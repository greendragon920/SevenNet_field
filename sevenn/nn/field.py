r"""
Electric-field coupling for SevenNet.

Design (see README_FIELD.md for the reasoning):

  A uniform external field E is expanded in real spherical harmonics Y_lm(E) with
  l >= 1 and injected into the SELF-CONNECTION (residual) branch of a convolution
  layer, as a purely additive term:

      self_cont_tmp  =  Linear(x)  +  TP(x, Y_lm(E))
                        \_pretrained_/   \__new, l>=1 only__/

  Two properties follow, and both are relied on elsewhere:

  1. Zero-field exactness.  Y_lm is a homogeneous polynomial of degree l in E,
     so for l >= 1 it vanishes identically at E = 0.  A tensor product is
     bilinear, hence the new term is *exactly* 0.0 there, and the electric
     enthalpy at E = 0 is bit-identical to the field-free energy no matter what
     the new weights are.  l = 0 is deliberately excluded: Y_00(E) = 1 even at
     E = 0, which would break this.

     Forces and stress are mathematically unchanged for the same reason, but
     they are NOT bit-reproducible in practice -- multithreaded CPU autograd
     already makes plain SevenNet's forces vary by ~1e-6 eV/A run to run in
     float32. Test them against that noise floor, not with torch.equal.
  2. Purely additive parameters.  No pretrained tensor changes shape, so a
     field-free checkpoint loads with strict=False and only the new modules'
     weights are missing.

  All dielectric observables are then derivatives of the single scalar enthalpy,
  which is what makes them mutually consistent (Maxwell reciprocity, the
  acoustic sum rule, and the crystal point-group symmetry of Z* and eps all hold
  by construction rather than by fitting).

Units:  E in V/Angstrom, energy in eV.  Then
        dipole    = -dF/dE          [e * Angstrom]
        P         = dipole / volume [e / Angstrom^2]
        Z*        = d(dipole)/dr    [e, dimensionless]
        chi       = d(dipole)/dE / (volume * eps0)   [dimensionless] = eps_inf - 1
"""

from typing import List, Optional

import torch
import torch.nn as nn
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, SphericalHarmonics

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .util import broadcast

# vacuum permittivity in e / (V * Angstrom)
EPS0_E_PER_V_ANGSTROM = 8.8541878128e-12 / 1.602176634e-19 / 1e10


def field_irreps(field_lmax: int) -> Irreps:
    """
    Irreps of Y_lm(E) for l = 1 .. field_lmax, with natural parity p = (-1)**l.

    l = 0 is excluded on purpose; see the module docstring.
    """
    assert field_lmax >= 1, 'field_lmax must be >= 1 (l=0 is never used)'
    return Irreps([(1, (ll, (-1) ** ll)) for ll in range(1, field_lmax + 1)])


class ElectricFieldPrepare(nn.Module):
    """
    Make sure the graph carries a uniform electric field of shape (n_graph, 3)
    that requires grad, so every downstream derivative w.r.t. E is available.

    If the field is absent it defaults to zero, which reproduces the field-free
    model exactly. That is the intended "off" state for correctness -- but note
    it does not save any compute, since the derivative machinery still runs. Use
    ``set_compute_field_response(False)`` on the model for that.
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
        data[self.key_field] = field.requires_grad_(True)
        return data


class ElectricFieldSelfConnection(nn.Module):
    """
    Add  TP(x, Y_lm(E))  into the self-connection buffer of one convolution layer.

    Must be placed immediately after ``{t}_self_connection_intro``, which is what
    creates the buffer this module accumulates into.

    At layer 0 the node features are scalars only, so the coupling rules allow
    exactly two paths, 0e x 1o -> 1o and 0e x 2e -> 2e. Those write into the two
    output blocks that the pretrained Linear leaves structurally zero (a linear
    map cannot make a vector out of a scalar), so nothing is displaced.
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_out: Irreps,
        field_lmax: int = 2,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_field: str = KEY.ELECTRIC_FIELD,
        data_key_out: str = KEY.SELF_CONNECTION_TEMP,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.key_x = data_key_x
        self.key_field = data_key_field
        self.key_out = data_key_out
        self._is_batch_data = True

        self.irreps_x = Irreps(irreps_x)
        self.irreps_out = Irreps(irreps_out)
        self.irreps_field = field_irreps(field_lmax)

        # normalize=False is mandatory: the MAGNITUDE of E is physical, unlike an
        # edge direction where only the angle matters. It also keeps Y smooth at
        # E = 0, where all the observables are evaluated.
        self.spherical = SphericalHarmonics(
            self.irreps_field, normalize=False, normalization='component'
        )
        self.tp = FullyConnectedTensorProduct(
            self.irreps_x, self.irreps_field, self.irreps_out
        )
        if zero_init:
            with torch.no_grad():
                for p in self.tp.parameters():
                    p.zero_()

    def num_paths(self) -> int:
        return len(self.tp.instructions)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        field = data[self.key_field]
        if self._is_batch_data:
            per_atom_field = field[data[KEY.BATCH]]
        else:
            per_atom_field = field.expand(x.shape[0], 3)
        y = self.spherical(per_atom_field)
        data[self.key_out] = data[self.key_out] + self.tp(x, y)
        return data


class FieldResponseOutput(nn.Module):
    """
    Differentiate the electric enthalpy to get polarization, Born effective
    charges and the electronic polarizability.

        dipole            = -dF/dE                       (n_graph, 3)
        P                 = dipole / volume              (n_graph, 3)
        Z*_{i,ab}         = d(dipole_a)/dr_{ib}          (n_atom, 3, 3)
        chi_{ab}          = d(dipole_a)/dE_b / (V eps0)  (n_graph, 3, 3)

    The position derivative is taken with respect to ``edge_vec`` rather than
    ``pos``, matching how SevenNet computes forces (``pos`` is not in the
    autograd graph, since edge vectors are built at graph-construction time).
    A useful side effect: edge vectors are invariant under a rigid translation,
    so the acoustic sum rule  sum_i Z*_i = 0  holds exactly, not approximately.

    Place this BEFORE ``force_output``: it retains the graph, whereas
    ``force_output`` in eval mode frees it.
    """

    def __init__(
        self,
        data_key_energy: str = KEY.PRED_TOTAL_ENERGY,
        data_key_field: str = KEY.ELECTRIC_FIELD,
        data_key_edge: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
        compute_bec: bool = True,
        compute_polarizability: bool = True,
    ) -> None:
        super().__init__()
        self.key_energy = data_key_energy
        self.key_field = data_key_field
        self.key_edge = data_key_edge
        self.key_edge_idx = data_key_edge_idx
        self.key_cell_volume = data_key_cell_volume
        self.compute_bec = compute_bec
        self.compute_polarizability = compute_polarizability
        self.enabled = True
        self._is_batch_data = True

    def _volume(self, data: AtomGraphDataType) -> torch.Tensor:
        vlim = 1e-3  # cell volume is 0 for non-PBC structures
        volume = data[self.key_cell_volume]
        volume = volume.reshape(-1).clone()
        volume[volume < vlim] = vlim
        return volume

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if not self.enabled:
            return data

        field = data[self.key_field]
        rij = data[self.key_edge]
        edge_idx = data[self.key_edge_idx]
        n_atom = int(torch.sum(data[KEY.NUM_ATOMS]).item())
        energy = data[self.key_energy].sum()

        # first derivative. create_graph=True because we differentiate it again.
        grad_field = torch.autograd.grad(
            [energy], [field], create_graph=True, retain_graph=True,
            allow_unused=True,
        )[0]
        if grad_field is None:
            grad_field = torch.zeros_like(field)
        dipole = torch.neg(grad_field)  # = volume * P, in e * Angstrom

        volume = self._volume(data)
        data[KEY.PRED_DIPOLE] = dipole
        data[KEY.PRED_POLARIZATION] = dipole / volume.unsqueeze(-1)

        need_second = self.compute_bec or self.compute_polarizability
        if not need_second:
            return data

        bec_cols: List[torch.Tensor] = []
        chi_cols: List[torch.Tensor] = []
        for a in range(3):
            wrt: List[torch.Tensor] = []
            if self.compute_bec:
                wrt.append(rij)
            if self.compute_polarizability:
                wrt.append(field)
            grads = torch.autograd.grad(
                [dipole[:, a].sum()],
                wrt,
                create_graph=self.training,
                retain_graph=True,
                allow_unused=True,
            )
            cursor = 0
            if self.compute_bec:
                g_edge: Optional[torch.Tensor] = grads[cursor]
                cursor += 1
                if g_edge is None:
                    bec_cols.append(
                        torch.zeros(n_atom, 3, dtype=rij.dtype, device=rij.device)
                    )
                else:
                    # d(scalar)/dr_k = (scatter over edge_idx[1]) - (over [0]),
                    # because edge_vec = pos[edge_idx[1]] - pos[edge_idx[0]].
                    # Mirrors ForceStressOutputFromEdge, with the sign that makes
                    # forces = pf - nf equal -dU/dr.
                    pf = torch.zeros(
                        n_atom, 3, dtype=g_edge.dtype, device=g_edge.device
                    )
                    nf = torch.zeros(
                        n_atom, 3, dtype=g_edge.dtype, device=g_edge.device
                    )
                    pf.scatter_reduce_(
                        0, broadcast(edge_idx[0], g_edge, 0), g_edge, reduce='sum'
                    )
                    nf.scatter_reduce_(
                        0, broadcast(edge_idx[1], g_edge, 0), g_edge, reduce='sum'
                    )
                    bec_cols.append(nf - pf)
            if self.compute_polarizability:
                g_field: Optional[torch.Tensor] = grads[cursor]
                if g_field is None:
                    chi_cols.append(torch.zeros_like(field))
                else:
                    chi_cols.append(g_field)

        if self.compute_bec:
            # stack along the polarization-component axis -> (n_atom, 3, 3),
            # index order [atom, polarization direction a, position direction b]
            data[KEY.PRED_BEC] = torch.stack(bec_cols, dim=1)
        if self.compute_polarizability:
            chi = torch.stack(chi_cols, dim=1)  # (n_graph, 3, 3)
            data[KEY.PRED_POLARIZABILITY] = (
                chi / volume.view(-1, 1, 1) / EPS0_E_PER_V_ANGSTROM
            )
        return data
