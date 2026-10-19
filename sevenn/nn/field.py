r"""
Electric-field coupling for SevenNet (field_physical: explicit electric enthalpy).

The backbone is left as it is and gives U(r). Heads give atomic dipoles and
polarizabilities (field_head: 'linear' and 'gate' read the node features that
enter the last convolution layer, the last ones that still carry vectors;
'bond' builds them from the final scalar features along the edges;
'bond_multi' does the same from the 0e features of every interaction block):

    mu_i     atomic dipole, 1o                         [e * Angstrom]
    alpha_i  atomic polarizability, sum_k v_ik v_ik^T + softplus(s_i) I
             (v_ik: 1o, s_i: 0e), positive semi-definite [e * Angstrom^2 / V]

and the electric enthalpy is the linear-response expansion

    F(r, E) = U(r) - E . M(r) - 1/2 E . alpha(r) . E,
    M = sum_i mu_i,  alpha = sum_i alpha_i.

Units: E in V/Angstrom, energy in eV.  Then
    dipole = -dF/dE = M + alpha . E              [e * Angstrom]
    P      = dipole / volume                     [e / Angstrom^2]
    Z*     = d(dipole)/dr                        [e]
    chi    = alpha / (volume * eps0) >= 0        [dimensionless] = eps_inf - 1
Forces at finite E come from differentiating F through the usual force output.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.nn import Gate
from e3nn.o3 import Irreps, Linear, SphericalHarmonics

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

# vacuum permittivity in e / (V * Angstrom)
EPS0_E_PER_V_ANGSTROM = 8.8541878128e-12 / 1.602176634e-19 / 1e10


class ElectricFieldPrepare(nn.Module):
    """
    Put a uniform electric field of shape (n_graph, 3) on the graph (zeros if absent).
    """

    def __init__(
        self,
        data_key_field: str = KEY.ELECTRIC_FIELD,
        data_key_reference: str = KEY.EDGE_VEC,  # only used for dtype/device
    ) -> None:
        super().__init__()
        self.key_field = data_key_field
        self.key_reference = data_key_reference
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        ref = data[self.key_reference]
        if self.key_field in data:
            field = data[self.key_field]
            field = field.detach().to(dtype=ref.dtype, device=ref.device)
            field = field.reshape(-1, 3)
        else:
            if self._is_batch_data:
                n_graph = int(data[KEY.BATCH].max().item()) + 1
            else:
                n_graph = 1
            field = torch.zeros(
                (n_graph, 3), dtype=ref.dtype, device=ref.device
            )
        data[self.key_field] = field
        return data


class FieldTensorHeads(nn.Module):
    """
    Read the atomic dipole mu_i and the atomic polarizability alpha_i from the
    node features entering the last convolution layer. Placed right before
    '{last}_self_connection_intro'; does not change the node features.
    """

    def __init__(
        self,
        irreps_x: Irreps,
        num_alpha_vectors: int = 4,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_dipole: str = KEY.ATOMIC_DIPOLE,
        data_key_polarizability: str = KEY.ATOMIC_POLARIZABILITY,
    ) -> None:
        super().__init__()
        self.key_x = data_key_x
        self.key_dipole = data_key_dipole
        self.key_polarizability = data_key_polarizability
        self.irreps_x = Irreps(irreps_x)
        self.num_alpha_vectors = num_alpha_vectors

        self.dipole_linear = Linear(self.irreps_x, Irreps('1x1o'))
        self.alpha_vector_linear = Linear(
            self.irreps_x, Irreps(f'{num_alpha_vectors}x1o')
        )
        self.alpha_scalar_linear = Linear(self.irreps_x, Irreps('1x0e'))

        # e3nn basis of a 1o feature -> Cartesian (x, y, z); taken from the same
        # spherical harmonics the edge embedding uses, so the convention matches.
        sph = SphericalHarmonics(
            Irreps('1x1o'), normalize=False, normalization='component'
        )
        basis = sph(torch.eye(3)) / math.sqrt(3.0)  # row i = Y1(e_i)
        self.register_buffer('to_cartesian', basis.T.contiguous())

    def num_paths(self) -> int:
        return len(self.dipole_linear.instructions)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        to_cartesian = self.to_cartesian.to(dtype=x.dtype)
        mu = self.dipole_linear(x) @ to_cartesian
        vectors = self.alpha_vector_linear(x).reshape(
            -1, self.num_alpha_vectors, 3
        ) @ to_cartesian
        isotropic = F.softplus(self.alpha_scalar_linear(x))  # (n_atom, 1)
        eye = torch.eye(3, dtype=x.dtype, device=x.device)
        alpha = torch.einsum('nka,nkb->nab', vectors, vectors) + isotropic.unsqueeze(
            -1
        ) * eye
        data[self.key_dipole] = mu
        data[self.key_polarizability] = alpha
        return data


def _sph_to_cartesian() -> torch.Tensor:
    """Matrix taking an e3nn 1o feature to Cartesian (x, y, z): v = f @ M."""
    sph = SphericalHarmonics(Irreps('1x1o'), normalize=False, normalization='component')
    return (sph(torch.eye(3)) / math.sqrt(3.0)).T.contiguous()


class FieldGateHeads(nn.Module):
    """
    Like FieldTensorHeads, with one gated hidden layer in between, so the
    heads can form scalar x vector products (e.g. a charge times a direction):
    x -> Linear -> Gate(silu scalars, sigmoid gates on 1o) -> Linear -> heads.
    """

    def __init__(
        self,
        irreps_x: Irreps,
        num_alpha_vectors: int = 4,
        hidden_multiplicity: int = 16,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_dipole: str = KEY.ATOMIC_DIPOLE,
        data_key_polarizability: str = KEY.ATOMIC_POLARIZABILITY,
    ) -> None:
        super().__init__()
        self.key_x = data_key_x
        self.key_dipole = data_key_dipole
        self.key_polarizability = data_key_polarizability
        self.irreps_x = Irreps(irreps_x)
        self.num_alpha_vectors = num_alpha_vectors
        h = hidden_multiplicity
        self.gate = Gate(
            Irreps(f'{h}x0e'), [F.silu],
            Irreps(f'{h}x0e'), [torch.sigmoid],
            Irreps(f'{h}x1o'),
        )
        self.hidden_linear = Linear(self.irreps_x, self.gate.irreps_in)
        irreps_hidden = self.gate.irreps_out
        self.dipole_linear = Linear(irreps_hidden, Irreps('1x1o'))
        self.alpha_vector_linear = Linear(irreps_hidden, Irreps(f'{num_alpha_vectors}x1o'))
        self.alpha_scalar_linear = Linear(irreps_hidden, Irreps('1x0e'))
        self.register_buffer('to_cartesian', _sph_to_cartesian())

    def num_paths(self) -> int:
        return len(self.hidden_linear.instructions)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        hidden = self.gate(self.hidden_linear(x))
        to_cartesian = self.to_cartesian.to(dtype=x.dtype)
        mu = self.dipole_linear(hidden) @ to_cartesian
        vectors = self.alpha_vector_linear(hidden).reshape(
            -1, self.num_alpha_vectors, 3
        ) @ to_cartesian
        isotropic = F.softplus(self.alpha_scalar_linear(hidden))
        eye = torch.eye(3, dtype=x.dtype, device=x.device)
        data[self.key_dipole] = mu
        data[self.key_polarizability] = torch.einsum(
            'nka,nkb->nab', vectors, vectors
        ) + isotropic.unsqueeze(-1) * eye
        return data


class FieldFeatureTap(nn.Module):
    """Copies the 0e channels of the current node features to data[data_key_out]
    (placed after an interaction block, so a bond head can read every block)."""

    def __init__(self, irreps_x: Irreps, data_key_out: str) -> None:
        super().__init__()
        self.key_out = data_key_out
        self.irreps_x = Irreps(irreps_x)
        self.scalar_index = torch.cat([
            torch.arange(sl.start, sl.stop)
            for (mul, ir), sl in zip(self.irreps_x, self.irreps_x.slices())
            if ir.l == 0 and ir.p == 1
        ]).tolist()

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_out] = data[KEY.NODE_FEATURE][:, self.scalar_index]
        return data


class FieldBondHeads(nn.Module):
    """
    Bond-charge / bond-polarizability heads on the final scalar node features:

        mu_i    = sum_j q_ij r_ij
        alpha_i = sum_j c_ij^2 rhat_ij rhat_ij^T + softplus(s_i) I

    q_ij = sum_b e_ij,b g_b(h_i, h_j) and c_ij likewise, with e_ij the radial
    edge embedding (it carries the cutoff function, so both vanish smoothly at
    the cutoff) and g a small MLP of the two atoms' scalar features. Placed
    right after the last convolution layer, before the energy readout.
    """

    def __init__(
        self,
        irreps_x: Irreps,
        num_basis: int,
        hidden_neurons: int = 64,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_dipole: str = KEY.ATOMIC_DIPOLE,
        data_key_polarizability: str = KEY.ATOMIC_POLARIZABILITY,
        tap_keys: List[str] = [],
        tap_dims: List[int] = [],
    ) -> None:
        super().__init__()
        self.key_x = data_key_x
        self.tap_keys = list(tap_keys)
        self.key_dipole = data_key_dipole
        self.key_polarizability = data_key_polarizability
        self.irreps_x = Irreps(irreps_x)
        scalar_slices = [
            sl for (mul, ir), sl in zip(self.irreps_x, self.irreps_x.slices())
            if ir.l == 0 and ir.p == 1
        ]
        assert scalar_slices, f'no 0e features in {self.irreps_x}'
        self.scalar_index = torch.cat(
            [torch.arange(sl.start, sl.stop) for sl in scalar_slices]
        ).tolist()
        n_scalar = len(self.scalar_index) + sum(tap_dims)
        self.num_basis = num_basis
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * n_scalar, hidden_neurons),
            nn.SiLU(),
            nn.Linear(hidden_neurons, 2 * num_basis),
        )
        self.alpha_scalar_linear = nn.Linear(n_scalar, 1)

    def num_paths(self) -> int:
        return len(self.scalar_index)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        h = data[self.key_x][:, self.scalar_index]
        if self.tap_keys:
            h = torch.cat([data[k] for k in self.tap_keys] + [h], dim=-1)
        edge_idx = data[KEY.EDGE_IDX]
        rij = data[KEY.EDGE_VEC]
        embedding = data[KEY.EDGE_EMBEDDING]
        src, dst = edge_idx[0], edge_idx[1]
        g = self.pair_mlp(torch.cat([h[src], h[dst]], dim=-1)).reshape(
            -1, 2, self.num_basis
        )
        q = (g[:, 0] * embedding).sum(-1)
        c = (g[:, 1] * embedding).sum(-1)
        length = torch.linalg.norm(rij, dim=-1, keepdim=True)
        rhat = rij / length
        n_atom = h.shape[0]
        mu = torch.zeros(n_atom, 3, dtype=rij.dtype, device=rij.device).index_add(
            0, src, q.unsqueeze(-1) * rij
        )
        bond_alpha = (c * c).view(-1, 1, 1) * rhat.unsqueeze(-1) * rhat.unsqueeze(-2)
        alpha = torch.zeros(
            n_atom, 3, 3, dtype=rij.dtype, device=rij.device
        ).index_add(0, src, bond_alpha)
        isotropic = F.softplus(self.alpha_scalar_linear(h))
        eye = torch.eye(3, dtype=rij.dtype, device=rij.device)
        data[self.key_dipole] = mu
        data[self.key_polarizability] = alpha + isotropic.unsqueeze(-1) * eye
        return data


class FieldEnthalpyOutput(nn.Module):
    """
    Add -E.mu_i - 1/2 E.alpha_i.E to the atomic and total energies, and write
    dipole, P, chi (direct) and Z* (one derivative of the dipole). Placed right
    before 'force_output', so forces include the field terms.
    """

    def __init__(
        self,
        data_key_field: str = KEY.ELECTRIC_FIELD,
        data_key_edge: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
        compute_bec: bool = True,
    ) -> None:
        super().__init__()
        self.key_field = data_key_field
        self.key_edge = data_key_edge
        self.key_edge_idx = data_key_edge_idx
        self.key_cell_volume = data_key_cell_volume
        self.compute_bec = compute_bec
        self._is_batch_data = True

    def _volume(self, data: AtomGraphDataType) -> torch.Tensor:
        vlim = 1e-3  # cell volume is 0 for non-PBC structures
        volume = data[self.key_cell_volume]
        volume = volume.reshape(-1).clone()
        volume[volume < vlim] = vlim
        return volume

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        field = data[self.key_field]
        mu = data[KEY.ATOMIC_DIPOLE]
        alpha = data[KEY.ATOMIC_POLARIZABILITY]
        n_atom = mu.shape[0]
        n_graph = field.shape[0]
        if self._is_batch_data:
            batch = data[KEY.BATCH]
        else:
            batch = torch.zeros(n_atom, dtype=torch.long, device=mu.device)

        atom_field = field[batch]
        induced = torch.einsum('nab,nb->na', alpha, atom_field)
        atomic_field_energy = -(atom_field * mu).sum(-1) - 0.5 * (
            atom_field * induced
        ).sum(-1)
        data[KEY.ATOMIC_ENERGY] = data[KEY.ATOMIC_ENERGY] + atomic_field_energy.view(
            -1, 1
        )
        graph_field_energy = torch.zeros(
            n_graph, dtype=mu.dtype, device=mu.device
        ).index_add(0, batch, atomic_field_energy)
        if self._is_batch_data:
            data[KEY.PRED_TOTAL_ENERGY] = (
                data[KEY.PRED_TOTAL_ENERGY] + graph_field_energy
            )
        else:
            data[KEY.PRED_TOTAL_ENERGY] = (
                data[KEY.PRED_TOTAL_ENERGY] + graph_field_energy.sum()
            )

        dipole = torch.zeros(
            n_graph, 3, dtype=mu.dtype, device=mu.device
        ).index_add(0, batch, mu + induced)
        alpha_graph = torch.zeros(
            n_graph, 3, 3, dtype=mu.dtype, device=mu.device
        ).index_add(0, batch, alpha)
        volume = self._volume(data)
        data[KEY.PRED_DIPOLE] = dipole
        data[KEY.PRED_POLARIZATION] = dipole / volume.unsqueeze(-1)
        data[KEY.PRED_SUSCEPTIBILITY] = (
            alpha_graph / volume.view(-1, 1, 1) / EPS0_E_PER_V_ANGSTROM
        )

        if not self.compute_bec:
            return data

        rij = data[self.key_edge]
        edge_idx = data[self.key_edge_idx]
        bec_cols: List[torch.Tensor] = []
        for a in range(3):
            g_edge = torch.autograd.grad(
                [dipole[:, a].sum()],
                [rij],
                create_graph=self.training,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if g_edge is None:
                bec_cols.append(
                    torch.zeros(n_atom, 3, dtype=rij.dtype, device=rij.device)
                )
                continue
            # same scatter as ForceStressOutputFromEdge (rij = r_j - r_i)
            pf = torch.zeros(n_atom, 3, dtype=g_edge.dtype, device=g_edge.device)
            nf = torch.zeros(n_atom, 3, dtype=g_edge.dtype, device=g_edge.device)
            pf.index_add_(0, edge_idx[0], g_edge)
            nf.index_add_(0, edge_idx[1], g_edge)
            bec_cols.append(nf - pf)
        # index order [atom, polarization a, position b]
        data[KEY.PRED_BEC] = torch.stack(bec_cols, dim=1)
        return data
