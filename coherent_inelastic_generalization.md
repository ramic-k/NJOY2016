# Generalized Coherent Inelastic Scattering in LEAPR

Implementation documentation for the generalized coherent inelastic treatment
in `leapr_generalized_inelastic_euphonics.py`. Enabled by setting `ncoh_inel=1`
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

Standard LEAPR computes S(alpha, beta) using the incoherent approximation: a
single isotropic phonon density of states (DOS) is fed through the phonon
expansion (`start` / `convol` / `contin`), producing only the self-scattering
contribution. This has two limitations:

1. **No coherent one-phonon structure.** The distinct (cross-atom) contribution
   to one-phonon scattering is absent. For materials with large coherent
   scattering cross sections (e.g., graphite, MgO, NaCl), the distinct
   contribution can be comparable to or larger than the self contribution.

2. **Isotropic DOS assumption.** A single DOS is used for all momentum transfer
   directions. For non-cubic crystals (e.g., graphite with hexagonal symmetry),
   the phonon DOS is anisotropic and a single isotropic average does not
   correctly represent the directional dependence of the phonon expansion.

The generalized treatment (`ncoh_inel=1`) addresses both by using Euphonic to
compute phonon eigenvectors and eigenfrequencies from force constants.

---

## 2. Theory

### 2.1 DOS tensor

For each species s in the unit cell, the partial phonon DOS is generalized to a
3x3 tensor:

```
rho_{s,ij}(omega) = (1/N_q) sum_{q,nu} Re(e*_{s,i}(q,nu) e_{s,j}(q,nu))
                    * delta(omega - omega_{q,nu}) / n_s
```

where `e_{s,i}(q,nu)` is the i-th Cartesian component of the mass-weighted
eigenvector for atoms of species s at wavevector q and branch nu, `N_q` is the
number of q-points in the Brillouin zone (BZ) mesh, and `n_s` is the number of
atoms of species s in the primitive cell.

The trace of this tensor recovers the standard isotropic partial DOS:

```
rho_iso(omega) = rho_{xx}(omega) + rho_{yy}(omega) + rho_{zz}(omega)
```

with `integral(rho_iso) = 3` (three translational degrees of freedom per atom).

For cubic crystals, `rho_{ij} = (rho_iso / 3) * delta_{ij}`, so the tensor is
diagonal and isotropic. For non-cubic crystals (hexagonal, tetragonal, etc.),
the off-diagonal and anisotropic diagonal components are non-zero.

### 2.2 Directional projection for the self-scattering phonon expansion

For a powder-averaging direction `khat`, the projected 1D DOS is:

```
rho(omega; khat) = sum_{ij} khat_i * rho_{ij}(omega) * khat_j
```

This projected DOS has the same normalization as the standard isotropic DOS
(integral = 1, since `|khat|=1` and the DOS tensor is normalized per degree of
freedom). It is passed directly to the existing `contin()` phonon expansion
routine, which computes the full multi-phonon series for that direction.

The directional self-scattering is then obtained by averaging `contin()` results
over `ndir` directions uniformly distributed on the sphere (using a golden
spiral sequence):

```
S_self(alpha, beta) = (1/ndir) * sum_{d=1}^{ndir} S_contin(alpha, beta; khat_d)
```

For cubic crystals, every direction gives the same DOS (rho_iso/3), so this
average reproduces the standard `contin()` result exactly. For non-cubic
crystals, the directional averaging captures the anisotropic phonon structure.

### 2.3 Debye-Waller factor

Each atom d in the unit cell has a 3x3 mean-square displacement (MSD) tensor
`W_d` (in Angstrom^2), computed by Euphonic from the force constants and
temperature. The Debye-Waller factor for momentum transfer Q is:

```
exp(-W_d(Q)) = exp(- sum_{ab} Q_a * W_{d,ab} * Q_b)
```

For cubic crystals, `W_d = (Tr(W_d)/3) * I`, recovering the standard isotropic
Debye-Waller factor `exp(-Q^2 * Tr(W_d)/3)`.

The connection to the LEAPR `dwpix` parameter (also called lambda_s or f0) is:

```
dwpix = 8 * pi^2 * (Tr(W_d)/3) * awr * kT / WL2EKIN
```

where `awr` is the atomic weight ratio, `kT` is the thermal energy in eV, and
`WL2EKIN = hbar^2 / (2 m_n) = 0.0818042 eV*Angstrom^2`.

### 2.4 One-phonon coherent scattering from BZ sampling

The one-phonon scattering amplitude for atom d at wavevector q and branch nu is:

```
A_d(q, nu) = (b_d / sqrt(m_d)) * exp(-W_d(Q)) * (Q . e*_d(q, nu)) * exp(i Q . r_d)
```

where `b_d` is the bound coherent scattering length, `m_d` is the atomic mass,
`e_d(q, nu)` is the phonon eigenvector, and `r_d` is the equilibrium position
in fractional coordinates.

The total one-phonon scattering intensity decomposes into self and distinct
parts. For a principal species P (the species whose TSL is being generated):

```
F_self_P(q, nu)    = sum_{d in P} |A_d(q, nu)|^2
F_distinct_P(q, nu) = F_total_P(q, nu) - F_self_P(q, nu)
```

where `F_total_P` depends on whether this species carries inter-species cross
terms (see Section 2.5 below).

Each mode is weighted by the Bose occupation factor and kinematic factor:

```
w(q, nu) = (n(omega) + 1) / (2 * N_atom * omega)
```

where `n(omega) = 1/(exp(hbar*omega/kT) - 1)` is the Bose-Einstein occupation
number.

For each alpha value (which maps to a momentum transfer magnitude
`|Q| = sqrt(alpha * awr * kT / WL2EKIN)`), phonon modes are computed at `ndir`
Q-vectors uniformly distributed on a sphere of radius `|Q|`. The self and
distinct contributions from all modes at all sphere points are histogrammed
onto the beta grid using linear interpolation, then averaged over the `ndir`
directions. This produces the raw BZ-sampled one-phonon arrays
`raw_self_grid(beta, alpha)` and `raw_distinct_grid(beta, alpha)`.

### 2.5 Inter-species cross terms and double-counting avoidance

For a compound material with multiple species (e.g., Na and Cl in NaCl),
separate TSL files are generated for each species. The transport code
reconstructs the total macroscopic scattering as:

```
Sigma(E -> E') = sum_s N_s * sigma_{b,s} * S_s(alpha, beta)
```

where `N_s` is the number density and `sigma_{b,s}` is the bound scattering
cross section of species s.

The coherent distinct one-phonon for species P involves interference between the
amplitudes of P atoms and all other atoms. Defining:

```
S_P   = sum_{d in P} A_d        (amplitude sum over principal atoms)
S_all = sum_{all d} A_d         (amplitude sum over all atoms)
```

the full one-phonon involving P is `Re(S_P* . S_all)`, which expands as:

```
Re(S_P* . S_all) = |S_P|^2 + Re(S_P* . S_other)
```

The first term `|S_P|^2` contains only intra-species P-P cross terms. The
second term `Re(S_P* . S_other)` is the inter-species cross term.

The inter-species cross term has the property that
`Re(S_P* . S_Q) = Re(S_Q* . S_P)` for any two species P and Q. If both species
include this term in their respective TSLs, it will be counted twice when the
transport code sums over all species.

To avoid this double-counting, the inter-species cross term is attributed
entirely to the species with the largest coherent scattering cross section.
For the species with the largest sigma_coh:

```
F_total_P = Re(S_P* . S_all)     [includes P-P and P-other cross terms]
```

For all other species:

```
F_total_P = |S_P|^2              [intra-species P-P cross terms only]
```

This convention parallels the treatment of coherent elastic scattering (CEF) in
polyatomic materials, where the Bragg diffraction cross section is attributed to
a single designated species.

For single-species materials (e.g., graphite), there are no inter-species terms
and `F_total_P = |S_all|^2` reduces to the standard formula.

### 2.6 Per-alpha cross-normalization

The BZ-sampled one-phonon arrays (`raw_self_grid`, `raw_distinct_grid`) are
computed in physical units (proportional to fm^2/amu) from the structure factor,
while the LEAPR phonon expansion operates in dimensionless LEAPR units. These
two unit systems differ by a normalization factor that depends on alpha.

To convert the BZ-sampled distinct one-phonon to LEAPR units, a per-alpha
cross-normalization is applied:

```
C(alpha) = sum_beta(ss1(beta, alpha)) / sum_beta(raw_self_grid(beta, alpha))
sd1(beta, alpha) = C(alpha) * raw_distinct_grid(beta, alpha)
```

where `ss1` is the self one-phonon computed from the isotropic DOS via
`start()` (in LEAPR units), and `raw_self_grid` is the BZ-sampled self
one-phonon (in physical units). The ratio `C(alpha)` converts from physical to
LEAPR units at each alpha independently.

Per-alpha normalization is used (rather than a single global normalization
factor) because the ratio between BZ-sampled and DOS-based self one-phonon can
vary with alpha. At low alpha (small Q), the BZ-sampled self includes coherent
Bragg-like peaks that are absent from the DOS-based self. A single global
normalization would be dominated by these low-alpha contributions and would
overestimate the distinct term at high alpha where it should be small.

### 2.7 Assembly of the total scattering law

The total S(alpha, beta) for the principal species P is:

```
S(alpha, beta) = S_self(alpha, beta) + (sigma_coh_P / sigma_b_P) * sd1(alpha, beta)
```

where:
- `S_self` is the directional powder-averaged phonon expansion (all phonon
  orders, computed by averaging `contin()` over ndir directions)
- `sd1` is the cross-normalized distinct one-phonon
- `sigma_coh_P = 4 * pi * b_coh_P^2` is the bound coherent cross section
- `sigma_b_P = spr * ((1 + awr) / awr)^2` is the bound scattering cross
  section (from the free cross section `spr`)

The self part `S_self` contains all phonon orders (1-phonon through nphon, plus
a short-collision-time Gaussian for the unconverged tail). The distinct part
`sd1` is one-phonon only; higher-order distinct contributions are not computed
because they require multi-dimensional convolutions that are not tractable in
the (alpha, beta) framework. In the incoherent approximation limit
(sigma_coh -> 0), the distinct term vanishes and the standard LEAPR result is
recovered.

---

## 3. Algorithm

### 3.1 Pre-loop (before temperatures)

1. Load force constants from phonopy via Euphonic (`ForceConstants.from_phonopy`)
2. Match Euphonic atoms to input Card 6d atom types by atomic number and mass
3. Compute 3x3 DOS tensors for all species from a single BZ mesh
   (`_precompute_all_dos_tensors_euphonic`)
4. Extract trace of principal species DOS tensor as the isotropic DOS
   (replaces Cards 6e, 11/12)

### 3.2 Per temperature

1. Compute isotropic `f0` (Debye-Waller lambda) and `tbar` (effective
   temperature ratio) from the isotropic DOS via `start()`
2. Compute per-atom 3x3 Debye-Waller tensors via Euphonic
   (`_compute_dw_euphonic`)
3. **Directional self-scattering**: For each of `ndir` golden-spiral
   directions, project the DOS tensor onto that direction via
   `rho(omega; khat) = khat^T . rho_tensor . khat`, run the full phonon
   expansion via `contin()`, accumulate the result. Parallelized across CPUs
   using `multiprocessing.Pool`.
4. **Distinct one-phonon**: For each alpha value, compute `|Q|` from alpha,
   generate `ndir` Q-vectors on a sphere, batch-compute phonon modes at those
   Q-points via Euphonic (`calculate_qpoint_phonon_modes`), evaluate the
   structure factor decomposition (Section 2.4-2.5), histogram onto the beta
   grid (`_vectorized_splat`), and apply per-alpha cross-normalization
   (Section 2.6).
5. Optionally smooth `sd1` along the beta axis with a Gaussian of width
   `sd1_sigma` meV (Section 4.2).
6. Assemble the total: `S = S_self + (sigma_coh / sigma_b) * sd1`

### 3.3 Call graph

```
run_leapr()
  _derive_spectra_from_phonopy()
    _precompute_all_dos_tensors_euphonic()     # BZ mesh -> DOS tensor per species
  temperature loop:
    compute_generalized_inelastic()
      start()                                  # isotropic f0, tbar from DOS
      _compute_dw_euphonic()                   # per-atom 3x3 DW tensor
      Pool.map(_contin_one_dir_worker, dirs)   # parallel directional contin()
        _project_dos() -> contin()
      compute_onephonon_eigvec_euphonic()      # BZ sphere sampling -> sd1
        golden_sphere()                        # ndir sphere points
        fc.calculate_qpoint_phonon_modes()     # Euphonic phonon solver
        structure factor decomposition         # F_self, F_distinct
        _vectorized_splat()                    # histogram onto beta grid
        per-alpha cross-normalization          # physical -> LEAPR units
        _gaussian_smooth()                     # optional smoothing
    assembly: S = S_self + weight * sd1
```

---

## 4. Input Format

### 4.1 Card 6b (extended)

```
elastic_mode  nat  nspec  ncoh_inel  /
```

- `ncoh_inel=0`: standard incoherent approximation (default)
- `ncoh_inel=1`: generalized coherent inelastic

### 4.2 Card 6f (read when ncoh_inel=1)

```
ndir  mesh_nx  mesh_ny  mesh_nz  nbin  ncpu  sd1_sigma  /
'phonopy_path'  /
'born_path'  /                         (optional)
```

| Parameter | Description | Default |
|---|---|---|
| `ndir` | Number of golden-spiral sphere directions used for both the directional self powder average and the per-alpha BZ sphere sampling | 500 |
| `mesh_nx/ny/nz` | Monkhorst-Pack BZ mesh per axis for DOS tensor computation | 30 30 30 |
| `nbin` | Number of energy bins for the DOS tensor | 300 |
| `ncpu` | Number of CPUs for parallel directional contin (0 = all available) | 0 |
| `sd1_sigma` | Gaussian smoothing width (meV) applied to sd1 along the beta axis before assembly. Suppresses statistical sampling noise from finite ndir without broadening physical features. 0 = no smoothing. | 0.0 |
| `phonopy_path` | Path to phonopy.yaml (with embedded force constants) or phonopy_disp.yaml (with FORCE_SETS in the same directory). Quoted string. | required |
| `born_path` | Path to BORN file for non-analytic corrections at the zone center. Auto-detected if present in the same directory as phonopy_path. | auto-detect |

With `ncoh_inel=1`, the input is simplified:

- **Cards 6e** (partial spectra): not needed; set `nspec=0`
- **Cards 11/12** (continuous DOS): not needed; derived from phonopy
- Cards 13+ (translational modes, discrete oscillators): still required if
  applicable

### 4.3 Choosing ndir

For the **directional self** part, convergence depends on crystal symmetry.
For cubic crystals the DOS is isotropic, so even `ndir=50` gives the exact
result. For non-cubic crystals (hexagonal graphite, etc.), `ndir=200-500` is
typically sufficient for the directional powder average to converge.

For the **distinct one-phonon**, `ndir` controls the number of Q-points sampled
on the sphere at each alpha. Statistical noise in sd1 decreases as
`1/sqrt(ndir)`. For low alpha (small Q), many phonon modes fall within the beta
range and the histogram is well-sampled even with modest ndir. For high alpha
(large Q), the Debye-Waller factor suppresses contributions and the sd1 signal
becomes small relative to the self contribution, so noise in sd1 has little
effect on the total.

The `sd1_sigma` parameter provides an alternative to increasing ndir: a
Gaussian filter of width 1-3 meV applied along beta suppresses the
high-frequency sampling noise without significantly broadening the physical
phonon features.

### 4.4 Example: graphite

```
leapr
20 /
'tsl C in graphite - ncoh_inel=1 Euphonic' /
1 0 100 /
28 6012 0 /
11.898 4.7392 1 10 0 0 /
0 0 0 0 0 /
1 1 0 1 /                              ! elastic_mode=1 nat=1 nspec=0 ncoh_inel=1
2.461 2.461 6.708 90.0 90.0 120.0 /
6 12 11.898 6.6460 0.001 4 /
0.0 0.0 0.0025  0.0 0.0 0.5025  0.333333 0.666667 0.0025  0.666667 0.333333 0.5025 /
500 40 40 40 300 0 0 /                  ! ndir mesh nbin ncpu sd1_sigma
'/path/to/graphite/phonopy.yaml' /
200 426 100 /
... alpha values .../
... beta values .../
296.0 /
0.0 0.0 1.0 /
0 /
```

### 4.5 Example: NaCl (Na TSL)

```
leapr
20 /
'Na in NaCl - ncoh_inel=1' /
1 0 100 /
100 11023 0 /
22.9898 3.28 1 10 0 0 /
0 0 0 0 0 /
2 2 0 1 /                              ! elastic_mode=2(MEF) nat=2 nspec=0 ncoh_inel=1
5.6903 5.6903 5.6903 90.0 90.0 90.0 /
11 23 22.9898 3.63 1.62 4 /
0.0 0.0 0.0  0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0 /
17 35 34.9689 11.65 4.70 4 /
0.5 0.5 0.5  0.5 0.0 0.0  0.0 0.5 0.0  0.0 0.0 0.5 /
100 10 10 10 200 0 2 /                 ! ndir mesh nbin ncpu sd1_sigma=2meV
'/path/to/NaCl/phonopy.yaml' /
366 367 0 /
... alpha values .../
... beta values .../
300.0 /
0.0 0.0 1.0 /
0 /
```

For NaCl, two separate runs are needed (one with `za=11023` for Na, one with
`za=17035` for Cl). The inter-species cross terms are attributed to Cl
(sigma_coh_Cl = 17.06 b > sigma_coh_Na = 1.66 b) as described in Section 2.5.

---

## 5. Implementation Details

### 5.1 Structure factor computation

The structure factor computation in `compute_onephonon_eigvec_euphonic()` is
fully vectorized over sphere directions and phonon modes. For a given alpha
value:

1. Compute `|Q| = sqrt(alpha * awr * kT / WL2EKIN)`
2. Generate `ndir` unit vectors on a sphere (golden spiral), scale to `|Q|`
3. Convert Cartesian Q-vectors to fractional coordinates via the inverse
   reciprocal lattice matrix
4. Batch-compute phonon modes at all Q-points via
   `fc.calculate_qpoint_phonon_modes(q_frac, asr='reciprocal')`
5. For all sphere points and modes simultaneously:
   - Compute per-atom Debye-Waller factors: `dw_d = exp(-Q . W_d . Q)`
   - Compute phase factors: `phase_d = exp(2*pi*i * q . r_d)`
   - Compute amplitudes: `A_d = (b_d/sqrt(m_d)) * dw_d * phase_d * (Q . e*_d)`
   - Decompose into self and distinct (see Section 2.4-2.5)
   - Weight by Bose factor and `1/(2*N_atom*omega)`
6. Histogram weighted contributions onto the beta grid using linear
   interpolation (`_vectorized_splat`)
7. Divide by ndir to average

The computation uses `numpy.einsum` for the tensor contractions and
`numpy.add.at` for the histogramming, avoiding Python loops over sphere points
or modes.

### 5.2 Gaussian smoothing of sd1

When `sd1_sigma > 0`, a 1D Gaussian filter of the specified width (in meV) is
applied independently to each alpha slice of `sd1(beta, alpha)`. The Gaussian
is truncated at 3 sigma and normalized to unit sum. The convolution preserves
the total integrated area of sd1 at each alpha.

The smoothing width is converted from meV to beta grid units using:
- For LAT=1: `sigma_beta = sd1_sigma / (THERM * 1000)` where THERM = 0.0253 eV
- Otherwise: `sigma_beta = sd1_sigma / (kT * 1000)`

then to grid-spacing units via `sigma_bins = sigma_beta / median(dbeta)`.

### 5.3 Phonopy as single source of truth

When `ncoh_inel=1`, all phonon data originates from a single set of force
constants:

| Data | Derived from | Replaces in standard LEAPR |
|------|-------------|----------|
| Partial DOS per species | DOS tensor trace | Card 6e spectra |
| Principal DOS | Principal species DOS tensor trace | Cards 11/12 |
| DW factors (elastic) | Euphonic DebyeWaller object | Card 6e dwpix |
| MSD tensors (inelastic DW) | Euphonic DebyeWaller object | Scalar MSD |
| Eigenvectors (distinct) | Euphonic phonon mode solver | N/A |

This guarantees physical consistency: the same force constants produce the DOS,
Debye-Waller factors, and eigenvectors for both elastic and inelastic channels.

---

## 6. Validation

### 6.1 Graphite vs FLASSH

Graphite (hexagonal, space group P6_3/mmc) was compared against a FLASSH
reference TSL at T = 296 K with identical alpha and beta grids (200 alpha, 426
beta). Both use LDA force constants (Euphonic from phonopy.yaml, FLASSH from
its own DFT).

The integrated S(Q,E) ratio (Euphonic / FLASSH) over the energy range
0.5-200 meV:

| Q range (1/Ang) | Ratio range | Notes |
|------------------|-------------|-------|
| 3 - 5 | 0.98 - 1.01 | Multiphonon dominated; both approaches converge |
| 5 - 10 | 0.95 - 0.99 | Good agreement |
| 10 - 13 | 0.92 - 0.97 | Slight underestimate due to DW factor difference |
| > 13 | 1.0 - 2.0 | Divergence from ~2% difference in Debye-Waller lambda between Euphonic LDA and FLASSH DFT force constants |

The shape comparison (area-normalized spectra) shows close agreement across
the full Q range, confirming that the spectral structure is correct and the
remaining differences are in the overall magnitude controlled by the
Debye-Waller factor.

At low Q (< 2 Ang^-1), the one-phonon coherent peaks differ between Euphonic
and FLASSH because these are sensitive to the specific phonon dispersion
details, which differ between the two sets of force constants.

### 6.2 Cubic NaCl isotropy check

NaCl (cubic, Fm-3m) was tested with separate Na and Cl TSLs. Because NaCl is
cubic, the directional self-scattering reproduces the isotropic result exactly:
`f0_avg / f0_iso = 1.000000` and `tbar_avg / tbar_iso = 1.000000` for both
species.

### 6.3 Inter-species cross term attribution

For NaCl, the inter-species cross term `Re(S_Na* . S_Cl)` is attributed
entirely to Cl (sigma_coh_Cl = 17.06 b > sigma_coh_Na = 1.66 b). This was
verified by comparing the Na TSL from the two-species run against a
"pseudo-NaCl" single-species run (where only Na atoms are treated as
scatterers and Cl atoms are invisible). Both produce identical sd1 sums,
confirming that Na's distinct one-phonon contains only Na-Na intra-species
cross terms.
