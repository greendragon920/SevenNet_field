# SevenNet-field (branch `field_1`)

Electric-field response for SevenNet: **polarization, Born effective charges (Z\*)
and electronic polarizability (χ = ε∞ − 1)**, obtained by differentiating a single
scalar electric enthalpy.

The design goal is to make a *pretrained* checkpoint field-aware without touching
any of its weights, so `7net-nano` can be fine-tuned on a BEC dataset without
risking its energy/force/stress accuracy.

---

## 1. What it does

A uniform external field `E` is expanded in real spherical harmonics `Y_lm(E)` with
**l ≥ 1** and injected into the **self-connection (residual) branch** of a
convolution layer as a purely additive bilinear term:

```
self_cont_tmp  =  Linear(x)   +   TP(x, Y_lm(E))
                  \pretrained/    \__new, l>=1 only__/
```

Every observable is then a derivative of one scalar, `F(r, E)`:

```
dipole      = -dF/dE                              [e Å]
P           = dipole / volume                     [e Å⁻²]
Z*_{i,ab}   = d(dipole_a)/dr_{ib}                 [e, dimensionless]
chi_{ab}    = d(dipole_a)/dE_b / (V eps0)         [dimensionless] = eps_inf - 1
```

Because they all come from the same scalar, **Maxwell reciprocity, the acoustic sum
rule Σᵢ Z\*ᵢ = 0, and the crystal point-group symmetry of Z\* and ε hold by
construction**, not by fitting.

### Why this placement

| property | mechanism |
|---|---|
| pretrained weights untouched | new module is additive; no existing tensor changes shape |
| **exact** zero-field identity | `Y_lm` is homogeneous of degree `l`, so for `l ≥ 1` it is *identically 0* at `E = 0`; a tensor product is bilinear, so the term is exactly `0.0` |
| uses genuinely free capacity | at layer 0 the pretrained `Linear` **cannot** reach the `1o`/`2e` output blocks (a linear map can't make a vector from a scalar) — those 256 of 352 outputs are structurally zero, and the field term fills exactly them |
| environment modulation for free | injected *before* the gate, so the field content is multiplied by an environment-dependent scalar gate |
| χ well-conditioned | including `Y_2(E)` supplies `E⊗E` explicitly, making χ **first order** in the new weights |

That last row is the main departure from MACE-Field, which injects `1o` only. MACE
can afford that because its *symmetric contraction* forms `A⊗A` products and can
build `E⊗E` itself. SevenNet has no product basis — node features never multiply
each other, the only nonlinearity is a scalar-mediated gate — so a `1o`-only
injection would leave χ reachable only through gate curvature. Supplying `Y_2(E)`
repairs that.

`l = 0` is deliberately **excluded**: `Y_00(E) = 1` even at `E = 0`, which would add
a permanently-on channel and destroy the zero-field guarantee. (Allegro-pol includes
it, so its zero-field model is a *different* model trained to agree, rather than
provably identical.)

---

## 2. Usage

```python
import sevenn._keys as KEY
from sevenn.model_build import build_E3_equivariant_model
import torch

cp  = torch.load('checkpoint_7net_nano_5.0.pth', map_location='cpu', weights_only=False)
cfg = dict(cp['config'])
cfg[KEY.USE_ELECTRIC_FIELD]      = True
cfg[KEY.FIELD_LMAX]              = 2      # Y_lm(E) for l = 1, 2
cfg[KEY.FIELD_INJECTION_LAYERS]  = [0]    # which conv layers to inject at

model = build_E3_equivariant_model(cfg)
model.load_state_dict(cp['model_state_dict'], strict=False)   # only new keys missing
model.set_is_batch_data(False)

data[KEY.ELECTRIC_FIELD] = torch.tensor([0.0, 0.0, 0.0])      # V/Angstrom
out = model(data)
out[KEY.PRED_BEC]              # (n_atom,  3, 3)
out[KEY.PRED_POLARIZABILITY]   # (n_graph, 3, 3), = eps_inf - 1
out[KEY.PRED_POLARIZATION]     # (n_graph, 3)
out[KEY.PRED_DIPOLE]           # (n_graph, 3)
```

Omitting `ELECTRIC_FIELD` defaults it to zero. Store it per graph as shape `(1, 3)`
in the dataset so PyG collation yields `(n_graph, 3)` — the same trick `STRESS` uses.

### Config

| key | default | meaning |
|---|---|---|
| `use_electric_field` | `False` | master switch |
| `field_lmax` | `2` | highest `l` of `Y_lm(E)`. `l = 0` is never used. |
| `field_injection_layers` | `[0]` | conv layers to inject at |

### Two different "off" switches

```python
data[KEY.ELECTRIC_FIELD] = zeros      # CORRECTNESS: identical to field-free nano
model.set_compute_field_response(False)   # SPEED: skips the double backward
```

Setting `E = 0` makes the output identical but still pays for the derivatives. Use
the second switch for plain MD with a field-aware checkpoint.

### Cost on 7net-nano-5.0

```
new parameters: 2,048  (+2.0% over the 106k backbone)
injection: 32x0e (x) 1x1o+1x2e -> 96x0e+32x1o+32x2e
  paths: 0e x 1o -> 1o  (1,024)
         0e x 2e -> 2e  (1,024)
```

---

## 3. What is verified

`claude_test_folder/test_field_model.py` (+ `.out`), run with the real pretrained
7net-nano-5.0 on rutile TiO₂ (periodic) and H₂O (non-periodic):

| check | result |
|---|---|
| field-free checkpoint loads, only `0_field_self_connection.tp.weight` missing | pass |
| **energy at E = 0 bit-identical** to plain nano (`torch.equal`) | pass |
| forces/stress at E = 0 within plain nano's own run-to-run floor | pass |
| varying field weights over 4 orders of magnitude leaves energy bit-identical | pass |
| Z\*, χ nonzero and correctly shaped | pass |
| acoustic sum rule Σᵢ Z\*ᵢ = 0 | 1.2e-07 |
| χ symmetric (Maxwell reciprocity) | 2.2e-08 |
| Z\* equivariant, `Z*(Rr) = R Z*(r) Rᵀ` | 5.4e-07 |
| dipole vs **central finite difference** of the energy | 7.6e-05 rel |
| dipole vanishes for centrosymmetric rutile (symmetry) | 2.6e-08 |
| gradients reach the field weights through the double backward | pass |

Regression: `pytest tests/unit_tests -k "not pretrained and not cueq and not flash
and not oeq and not d3"` gives **171 passed / 1 skipped / 0 failed**, identical to
the same run on `main`.

### On bit-identity, precisely

Only the **energy** is bit-reproducible. Forces and stress are mathematically
unchanged at `E = 0`, but *plain SevenNet's own* forces vary by ~5e-7 eV/Å between
identical repeated runs (multithreaded float32 autograd). So the test measures that
intrinsic floor and asserts the field model stays within it. Do not write
`torch.equal` assertions on forces — they fail for unmodified SevenNet too.

---

## 4. Not implemented yet

The **model** is complete and tested. The **training pipeline** is not:

- [ ] read `REF_becs` / `REF_polarizability` from extxyz (`train/dataload.py`,
      `train/graph_dataset.py`); needs per-atom `(N,3,3)` and per-graph `(1,3,3)`
      keys added to `atoms_to_graph`
- [ ] `BECLoss` / `PolarizabilityLoss` in `train/loss.py` + weights in config
- [ ] error-recorder entries in `error_recorder.py` so BEC/χ RMSE is logged
- [ ] backbone freezing option (currently freeze manually:
      `for n, p in model.named_parameters(): p.requires_grad_('field' in n)`)
- [ ] LAMMPS / TorchScript deploy — `patch_electric_field` raises
      `NotImplementedError` for `parallel=True`
- [ ] fused kernels: `cueq` / `oeq` / `flash_tp` almost certainly lack
      double-backward support, so they must be off whenever Z\* is requested

---

## 5. Known limitations

- **Polarization is unconstrained by a BEC-only dataset.** Z\* constrains only
  `∂(ΩP)/∂r`; the absolute `P` is never seen by the loss. Do not trust
  `PRED_POLARIZATION` after training on MP-Dielectrics alone — that is why the
  MACE-Field paper used a separate ferroelectric-path dataset with `P` labels.
- **Nothing constrains the surface beyond E².** The data are derivatives *at*
  E = 0. Finite-field MD at large |E| is extrapolation, and the `Y_2(E)` term grows
  as |E|². Monitor `‖field 2e block‖ / ‖conv 2e block‖` if you go there.
- **Element-only coupling at layer 0.** `x` is the raw element embedding there, so
  the bare field coupling is `c(Z) = W·e(Z)`, a shared linear map on the 32-dim
  embedding. Anisotropy and environment dependence are generated downstream by two
  convolutions plus the gate. Whether that reaches *anomalous* Z\* is the open
  empirical question — `field_injection_layers = [0, 1]` is the next rung (adds
  12,288 params and the `1o⊗1o→0e` path, i.e. an atom-dipole rather than
  bond-dipole picture).
- **Locality.** 3 message-passing layers at 5 Å is a local approximation to a
  partly delocalised electronic response. Shared with MACE-Field and Allegro-pol.
- **Requires odd-parity features.** `h ⊗ 1o → h` has zero paths if the backbone
  carries only even-parity irreps; `patch_electric_field` raises in that case.
  7net-nano and 7net-omni both carry `1o`, so both are fine.

---

## 6. Dataset notes (MP-Dielectrics)

From `claude_test_folder/step7_dataset_coverage.py`:

```
4,225 frames, 72,730 atoms, 80 distinct elements
in dataset but not in nano : none
in nano but not in dataset : 11  (He Ne Ar Kr Pm Yb Po At Pa Np Pu)
```

The 80 covered elements **span the full 32-dim embedding space**, and the coupling
is a shared linear map on that embedding rather than a per-element parameter, so all
11 missing elements have out-of-span component exactly 0 — they interpolate (Pm↔Sm
cosine 0.996, Np↔U 0.936, At↔I 0.854).

The real risk is elsewhere: **13 elements appear in fewer than 20 atoms total**
(Eu and Gd in *two* atoms each), and oxygen alone is 27% of all atoms. Report
per-element Z\* errors, not just a global RMSE, and consider inverse-frequency loss
weighting.

---

## 7. Files

```
sevenn/nn/field.py            ElectricFieldPrepare, ElectricFieldSelfConnection,
                              FieldResponseOutput           (new)
sevenn/model_build.py         patch_electric_field()        (+ parallel guard)
sevenn/nn/sequential.py       set_compute_field_response()
sevenn/_keys.py               ELECTRIC_FIELD, BEC, POLARIZABILITY, PRED_*,
                              USE_ELECTRIC_FIELD, FIELD_LMAX, FIELD_INJECTION_LAYERS
sevenn/_const.py              defaults + validation

claude_test_folder/test_field_model.py    the test suite for this change
claude_test_folder/step*.py               exploratory analyses behind the design
```

Module order after patching (nano, injection at layer 0):

```
electric_field_prepare        <- new, ensures E exists and requires grad
edge_embedding
onehot_idx_to_onehot
onehot_to_feature_x
0_self_connection_intro
0_field_self_connection       <- new, the injection
0_self_interaction_1
0_convolution
...
reduce_total_enegy
field_response                <- new, the derivatives
force_output
```

`field_response` sits **before** `force_output` because it retains the autograd
graph, whereas `force_output` frees it in eval mode.
