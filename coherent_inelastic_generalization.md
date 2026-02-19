# Generalized Coherent Inelastic Scattering

Design document for the coherent inelastic treatment in
`leapr_generalized_full_inelastic.py`. Enabled by setting `ncoh_inel=1`
on Card 6b.

**References:**

- K. Ramic et al., "NJOY+NCrystal", *NIM-A*, **1027** (2022) 166227.
- N. C. Fleming et al., "*FLASSH* 1.0: Thermal Scattering Law Evaluation and
  Cross-Section Generation for Reactor Physics Applications", *Nuclear Science
  and Engineering*, **197**(8) (2023) 1887-1901.
  DOI: 10.1080/00295639.2023.2194195
- Y. Q. Cheng and A. J. Ramirez-Cuesta, "Calculation of the Thermal Neutron
  Scattering Cross-Section of Solids Using OCLIMAX", *JCTC*, **16**(8) (2020)
  5212-5217. DOI: 10.1021/acs.jctc.0c00569
- Y. Q. Cheng et al., "Simulation of Inelastic Neutron Scattering Spectra
  Using OCLIMAX", *JCTC*, **15**(3) (2019) 1974-1982.
  DOI: 10.1021/acs.jctc.8b01250
- M. T. Garba, D. L. Roach, and H. Gonzalez-Velez, "Computational modelling
  and simulation of polycrystalline coherent inelastic neutron scattering",
  *Simulation Modelling Practice and Theory*, **77** (2017) 338-349.
  DOI: 10.1016/j.simpat.2017.08.001

---

## 1. Motivation

Standard LEAPR computes S(alpha, beta) using the **incoherent approximation**:
a single isotropic phonon DOS is fed through the phonon expansion (start /
convol / contin), producing only the self-scattering contribution. This has
two limitations:

1. **No coherent one-phonon.** The distinct (cross-atom) contribution to
   one-phonon scattering is absent. For materials with large coherent
   cross-sections (MgO, Al2O3, UO2), this matters.

2. **Isotropic DOS assumption.** A single DOS is used for all Q-directions.
   For non-cubic crystals (graphite, TiO2), the phonon DOS is anisotropic
   and an isotropic average treats this incorrectly.

The generalized treatment (`ncoh_inel=1`) fixes both by using phonopy
eigenvectors and lattice dynamics directly.

---

## 2. Physics

### 2.1 DOS tensor

For each atom species d, the partial DOS becomes a 3x3 tensor:

```
rho_{d,ij}(omega) = (1/N_q) sum_{q,nu} Re(e*_{d,i} e_{d,j}) delta(omega - omega_{q,nu}) / n_d
```

where `e_{d,i}(q,nu)` is the Cartesian eigenvector component. The trace
recovers the standard isotropic DOS: `Tr(rho) = rho_iso`, and
`integral(Tr(rho)) = 3` (three DOF per atom).

### 2.2 Directional projection

For a powder-averaging direction khat, the projected 1D DOS is:

```
rho(omega, khat) = khat . rho_tensor . khat
```

This has the same normalization as the standard DOS and feeds directly into
`contin()`.

### 2.3 MSD tensor and anisotropic Debye-Waller

Each species gets a 3x3 mean-square displacement tensor
`B_{d,ij} = <u_{d,i} u_{d,j}>`, computed from the DOS tensor. The DW factor
for momentum transfer Q becomes direction-dependent:

```
exp(-W_d) = exp(-1/2 Q . B_d . Q)
```

For cubic crystals: B = (msd/3) I, recovering the standard isotropic DW.

### 2.4 One-phonon distinct

The coherent cross-atom contribution uses structure factors from BZ sampling:

```
A_d(q,nu) = (b_d / sqrt(m_d)) * exp(-W_d(Q)) * (Q . e_d(q,nu)) * exp(i Q . r_d)

F_coherent = |sum_d A_d|^2     (total, including interference)
F_self     = sum_d |A_d|^2     (self only)
S_distinct = F_coherent - F_self
```

### 2.5 Final scattering law

```
S(alpha, beta) = S_self(alpha, beta)                              [directional phonon expansion]
               + (sigma_coh / sigma_b) * S_d^1(alpha, beta)      [one-phonon distinct]
```

---

## 3. Algorithm

### 3.1 Pre-loop (before temperatures)

1. Load phonopy from the path on Card 6f
2. Match phonopy atoms to Card 6d atom types
3. Compute DOS tensors for all species from a single BZ mesh
4. Extract trace as isotropic partial DOS per species (replaces Cards 6e, 11/12)

### 3.2 Per temperature

1. Compute scalar MSD per species (for elastic DW)
2. Compute isotropic f0 and tbar via `start()` (for DW lambda and T_eff)
3. Compute 3x3 MSD tensor per species from DOS tensor
4. **Directional self**: for ndir random directions, project DOS tensor onto
   each direction, run full phonon expansion via `contin()`, average results.
   Parallelized across CPUs with `multiprocessing.Pool`.
5. **One-phonon distinct**: for each alpha, sample ndir Q-directions in the BZ,
   compute coherent structure factors from phonopy eigenvectors, accumulate
   S_d^1 on the beta grid.
6. Combine: S_eff = S_self + weighted S_d^1

### 3.3 Call graph

```
main()
  _derive_spectra_from_phonopy()
    _precompute_all_dos_tensors()       # phonopy mesh -> DOS tensor per species
  temperature loop:
    compute_generalized_inelastic()
      start()                           # isotropic f0, tbar
      _compute_msd_tensor()             # 3x3 MSD per species
      Pool.map(contin_worker, dirs)     # parallel direction loop
        _project_dos() -> contin()
      compute_onephonon_eigvec()        # BZ sampling -> distinct S_d^1
```

---

## 4. Phonopy as Single Source of Truth

When `ncoh_inel=1`, ALL phonon data comes from phonopy:

| Data | Derived from | Replaces |
|------|-------------|----------|
| Partial DOS per species | DOS tensor trace | Card 6e spectra |
| Principal DOS | Principal species trace | Cards 11/12 |
| DW factors (elastic) | Partial spectra via start() | Card 6e DW |
| MSD tensors (inelastic) | DOS tensor integration | Scalar MSD |
| Eigenvectors (distinct) | Phonopy dynamical matrix | N/A |

This guarantees physical consistency: the same force constants produce the
DOS, DW, and eigenvectors for both elastic and inelastic channels.

---

## 5. Input Format

### Card 6b (extended)

```
elastic_mode  nat  nspec  ncoh_inel  /
```

- `ncoh_inel=0`: standard incoherent approximation (default)
- `ncoh_inel=1`: generalized coherent inelastic

### Card 6f (read when ncoh_inel=1)

```
ndir  mesh_nx  mesh_ny  mesh_nz  sigma  nbin  ncpu  /
'phonopy_disp_yaml_path'  /
'born_path'  /                         (optional)
```

| Parameter | Description | Default |
|---|---|---|
| `ndir` | Powder-averaging directions | 500 |
| `mesh_nx/ny/nz` | Phonopy BZ mesh per axis | 30 30 30 |
| `sigma` | Gaussian smearing width [THz] | 0.0 |
| `nbin` | DOS energy grid points | 300 |
| `ncpu` | CPUs for parallel directions (0 = all) | 0 |
| `phonopy_path` | Path to phonopy_disp.yaml | required |
| `born_path` | Path to BORN file | auto-detect |

### Simplified input

With `ncoh_inel=1`, the input is shorter:

- **Cards 6e** (partial spectra): not needed, set `nspec=0`
- **Cards 11/12** (continuous DOS): not needed, derived from phonopy
- Cards 13+ (translational, oscillators): still required

### Choosing ndir

Convergence goes as 1/sqrt(ndir). For **cubic** crystals the DOS is
isotropic, so even ndir=50 suffices for the self part (the distinct BZ
sampling still benefits from more directions). For **non-cubic** crystals,
ndir=500-2000 is recommended. Run with increasing ndir until S(alpha, beta)
stops changing.

### Example (MgO)

```
leapr
20
'tsl Mg in MgO'/
1 0 100/
58 12024 0/
23.785 3.631 1 10 0 0/
0 0 0 0 0/
1 2 0 1/                              ! elastic_mode, nat, nspec, ncoh_inel
4.2556 4.2556 4.2556 90.0 90.0 90.0/
12 24 23.785 5.375 0.08 4/
0.0 0.0 0.0  0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0/
8 16 15.858 5.803 0.0008 4/
0.5 0.5 0.5  0.5 0.0 0.0  0.0 0.5 0.0  0.0 0.0 0.5/
500 30 30 30 0.0 300 0/               ! ndir, mesh, sigma, nbin, ncpu
'/path/to/phonopy_disp.yaml'/
20 20 1/
... alpha values .../
... beta values .../
296.0
0.0 0.0 1.0/
0/
0/
stop
```

---

## 6. Validation

### Cubic MgO (Fm-3m)

MgO is cubic, so the generalized treatment must reduce to the standard
incoherent result for the self part. All isotropic limits are recovered:

| Metric | Expected | Measured |
|--------|----------|----------|
| DOS integral (Mg, O) | 3.0000 | 3.0000 |
| Tr(B) / scalar MSD | 1.000000 | 1.000000 |
| f0_directional / f0_isotropic | 1.000000 | 1.000000 |
| tbar_directional / tbar_isotropic | 1.000000 | 1.000000 |

The distinct contribution S_d^1 is nonzero (Mg-O cross terms), weighted by
sigma_coh / sigma_b = 0.92.

### Old vs new input format

Running with the old format (nspec=2, Cards 6e + 11/12 in the file) and the
new simplified format (nspec=0, phonopy-derived) produces **bit-identical**
ENDF output.

---

## 7. Design Decisions

| Decision | Rationale |
|---|---|
| Phonopy as single source | Consistency: same force constants for DOS, DW, and eigenvectors. Eliminates mismatched spectra. |
| multiprocessing (not threads) | `contin()` is CPU-bound Python; threads are GIL-limited. Processes give true parallelism. |
| 3-parameter mesh (nx, ny, nz) | Anisotropic BZ sampling for non-cubic crystals (e.g. 20 20 40 for hexagonal). |
| Backward-compatible parser | Old inputs (nspec>0) still work: Cards 6e/11/12 are read but overridden by phonopy values. |
| Self-consistent normalization C | Maps physical structure factor units (fm^2/amu) to LEAPR dimensionless using validated self one-phonon as reference. |
