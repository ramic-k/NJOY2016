# Generalized Coherent Elastic Scattering: Algorithm and Design

This document describes the physics, algorithm, and design of the generalized
coherent elastic treatment implemented in `leapr_generalized.py`. The code
handles arbitrary crystal structures (any symmetry, any number of atom species)
and produces ENDF-6 MF7/MT2 output in either Current ENDF Format (CEF) or
Mixed Elastic Format (MEF).

**References:**

- K. Ramic, J. I. Damian Marquez, et al., "NJOY+NCrystal: An open-source tool
  for creating thermal neutron scattering libraries with mixed elastic support",
  *Nuclear Instruments and Methods in Physics Research Section A*, **1027** (2022)
  166227.
- T. Kittelmann et al., "Elastic neutron scattering models for NCrystal",
  *Computer Physics Communications*, **267** (2021) 108082.

---

## 1. Physics Background

### 1.1 Coherent elastic cross-section for a powder

For a polycrystalline material, the coherent elastic cross-section per atom is
(Squires, *Introduction to the Theory of Thermal Neutron Scattering*):

```
σ_coh_el(E) = (λ² / 4 V_atom) · Σ_{d ≥ λ/2}  mult · |F(hkl)|² · d
```

where:

| Symbol | Meaning |
|---|---|
| `E` | Neutron kinetic energy [eV] |
| `λ = sqrt(WL2EKIN / E)` | Neutron wavelength [Å] |
| `WL2EKIN = 0.081804209605330899 eV·Å²` | ℏ²/(2 m_n), from NCrystal/CODATA 2018 |
| `V_atom = V_cell / N_atoms` | Volume per atom [Å³] |
| `d` | d-spacing of plane set (hkl) [Å] |
| `mult` | Crystallographic multiplicity |
| `F(hkl)` | Structure factor [√barn] |

Each plane set enters when E exceeds its Bragg threshold:

```
E_threshold(d) = WL2EKIN / (4 d²)
```

The cross-section is computed as a running cumulative sum divided by energy:

```
σ(E) = cumulative(E) / E   [barn]
```

where each plane contributes:

```
Δ = d · |F|² · mult · 0.5 · WL2EKIN / (V_cell · N_atoms)   [barn·eV]
```

### 1.2 Structure factor

For a unit cell with multiple atom species:

```
F(hkl) = Σ_s  b_s · f_s(hkl)
```

where `f_s(hkl) = Σ_{j ∈ species s} exp(i 2π (h·x_j + k·y_j + l·z_j))` is the
per-species geometric form factor, and `b_s` is the coherent scattering length
in √barn (`b_coh_fm / 10`).

The squared structure factor:

```
|F|² = |Σ_s b_s · f_s|² = Σ_{s,t} b_s b_t · Re[f_s · f_t*]
```

### 1.3 Debye-Waller factor — per-species treatment

The Bragg-edge calculator stores **temperature-independent** per-plane
contributions (DW = 1). The Debye-Waller attenuation is applied when building
the ENDF output, using **per-species** DW factors from partial phonon spectra.

For a single effective DW parameter `w`, the standard ENDF accumulation is:

```
cumulative += exp(-4 w E_j) · Δ_j
```

For per-species DW (as implemented), the DW is applied inside the species sum:

```
δ_j(T) = scale × Σ_{s,t} b_s · b_t · exp(-2(W_s + W_t) · E_j) · D_{st,j}
```

where:
- `W_s = dwpix_s / (awr_s × T × kB)` is the per-species ENDF DW parameter [1/eV]
- `dwpix_s` is the raw DW lambda from the partial phonon spectrum for species `s`
- `D_{st,j}` is the per-species correlation matrix (see Section 3)
- `scale` depends on the elastic mode (CEF or MEF)

This matches NCrystal's approach, where each atom type in the structure factor
gets its own DW exponent `exp(-W_i · q²/2)`. The result is ~0.03% agreement
with NCrystal reference data for SiC.

Note: when all species have the same W, the per-species formula reduces to
`exp(-4W·E) · Δ`, recovering the standard single-DW behavior exactly.

---

## 2. Per-Species Correlation Matrix

The Bragg-edge calculator (`compute_bragg_edges_general`) optionally returns a
per-species correlation matrix alongside the standard `bragg_data` array. This
is activated by `per_species=True`.

For each Bragg edge `j`, the correlation matrix is:

```
D_{st,j} = d_j · C_{st,j} · mult_j · xsectfact
```

where `C_{st}(hkl) = Re[f_s(hkl) · f_t*(hkl)]` is the geometric correlation
between species `s` and `t` (form factors WITHOUT scattering lengths).

The relationship to the standard output:

```
Δ_j = bragg_data[j, 1] = Σ_{s,t} b_s · b_t · D_{st,j}
```

The returned `species_corr` array has shape `(nbe, nspecies, nspecies)`.

---

## 3. Elastic Modes: CEF and MEF

### 3.1 CEF — Current ENDF Format (elastic_mode=1)

The CEF approach stores elastic scattering in the existing ENDF LTHR=1/2 format.
One atom is designated the **Dominant Channel (DC) atom** and carries the
coherent elastic Bragg edges; other atoms receive incoherent elastic with a
redistribution correction.

**Single atom (nat=1):**
- If σ_coh > σ_inc: LTHR=1 (coherent), scale = (σ_coh + σ_inc) / σ_coh
- If σ_inc > σ_coh: LTHR=2 (incoherent), scale = (σ_coh + σ_inc) / σ_inc

**Polyatomic (nat>1):**
- DC atom selection: minimize `f_i/(1-f_i) × σ_inc_i` over all atom types
- DC atom tape: LTHR=1, cumulative S scaled by `1/f_DC`
- Non-DC atom tapes: LTHR=2, SB = σ_inc × redistribution factor
  - Redistribution: `[1 + f_DC/(1-f_DC) × σ_inc_DC/σ_inc_i]`

### 3.2 MEF — Mixed Elastic Format (elastic_mode=2)

The MEF approach uses LTHR=3, which stores both coherent and incoherent elastic
in a single MF7/MT2 section. Every atom gets the same treatment:

- **Coherent part**: Bragg edges per atom (no DC scaling, shared across all atoms)
- **Incoherent part**: SB = σ_inc of the principal scatterer, Wp = per-species DW

For polyatomic materials, both atoms produce the **same** Bragg edge table
(since the coherent cross-section is per-atom, not per-species), but different
SB and Wp values.

---

## 4. Algorithm: Bragg-Edge Calculation

### 4.1 Overview

The `compute_bragg_edges_general()` function replaces the hardcoded `coher()`
subroutine. It handles any crystal symmetry and any number of atom species.
The algorithm follows NCrystal (`NCFillHKL.cc` + `NCPowderBragg.cc`).

### 4.2 Steps

1. **Reciprocal lattice matrix**: `G = _get_reciprocal_lattice_matrix(a, b, c,
   α, β, γ)` maps Miller indices to k-vectors. Analytic special cases for
   cubic/tetragonal/orthorhombic, hexagonal, and monoclinic; general triclinic
   falls back to `G = 2π · L⁻¹`.

2. **HKL enumeration**: NCrystal convention — enumerate the positive half of
   reciprocal space (h≥0; h=0→k≥0; h=k=0→l>0). Each (h,k,l) represents its
   Friedel pair, so initial multiplicity = 2.

3. **Structure factor**: For each (h,k,l), compute per-species form factors
   f_s and total F² = |Σ_s b_s · f_s|². Optionally compute the correlation
   matrix C_{st} = Re[f_s · f_t*].

4. **Multiplicity grouping**: Planes with the same d-spacing AND F² are merged,
   accumulating multiplicity. This naturally discovers crystallographic
   multiplicity for any symmetry.

5. **Energy assembly**: Convert d-spacing to E_threshold, multiply by xsectfact,
   sort ascending, merge near-degenerate energies.

6. **Emax sentinel**: Append a final entry at emax (LEAPR/NJOY convention).

### 4.3 Data structures

```python
@dataclass
class AtomSite:
    b_coh_fm: float                          # coherent scattering length [fm]
    positions: List[Tuple[float, float, float]]  # fractional coordinates

@dataclass
class CrystalStructure:
    a, b, c: float           # lattice constants [Å]
    alpha, beta, gamma: float  # lattice angles [degrees]
    sites: List[AtomSite]    # one per distinct species
```

---

## 5. Input Cards for Generalized Elastic (iel=10)

When `iel=10` on Card 5, additional cards are read after Card 6:

### Card 6b — Elastic mode and atom counts

```
elastic_mode  nat  nspec  /
```

| Parameter | Values |
|---|---|
| `elastic_mode` | 1 = CEF (LTHR=1/2), 2 = MEF (LTHR=3) |
| `nat` | Number of distinct atom types in the unit cell |
| `nspec` | Number of partial phonon spectra to follow |

### Card 6c — Lattice parameters

```
a  b  c  alpha  beta  gamma  /
```

Lattice constants in Å, angles in degrees.

### Card 6d — Atom types (repeated nat times)

```
Z  A  awr  b_coh  sigma_inc  npos  /
x1 y1 z1  x2 y2 z2  ...  /
```

| Field | Description |
|---|---|
| `Z, A` | Atomic and mass number |
| `awr` | Atomic weight ratio to neutron |
| `b_coh` | Coherent scattering length [fm] |
| `sigma_inc` | Incoherent cross section [barns] |
| `npos` | Number of positions in the unit cell |
| `x y z` | Fractional coordinates (npos × 3 values) |

### Card 6e — Partial phonon spectra (repeated nspec times)

```
Z  A  delta  ni  /
rho(1) rho(2) ... rho(ni)  /
```

Matched to atom types by (Z, A). Used to compute per-species DW factors.

---

## 6. Per-Species DW Computation

The `_compute_per_species_msd()` function computes the DW lambda for each atom
type from its partial phonon spectrum, using the same method as LEAPR's `start()`
function:

1. Transform phonon DOS: `p[j] = rho[j] / (β · 2sinh(β/2))`
2. Normalize: `p[j] /= fsum(1, p, ni, τ, Δβ)`
3. Compute DW lambda: `dwpix_s = fsum(0, p, ni, τ, Δβ)`

The ENDF DW parameter is then: `W_s = dwpix_s / (awr_s × T × kB)`

For SiC at 293.15 K:
- Si: dwpix = 0.889947, W = 1.265220 [1/eV]
- C:  dwpix = 0.463835, W = 1.541939 [1/eV]

The lighter carbon atom has a larger W (more thermal displacement) than the
heavier silicon atom, as expected physically.

---

## 7. ENDF Output Builders

### `_build_cef_coherent` — LTHR=1

Builds the coherent elastic Bragg edge table with per-species DW:

```python
def _edge_delta(j, itemp, energy=None):
    """DW-weighted edge contribution."""
    e = energy if energy is not None else bragg[j][0]
    delta = 0.0
    for si in range(nsp):
        for ti in range(nsp):
            dw = exp(-2.0 * (W_ps[si][itemp] + W_ps[ti][itemp]) * e)
            delta += b_sqb[si] * b_sqb[ti] * dw * species_corr[j, si, ti]
    return delta * scale
```

Falls back to single-DW `exp(-4w·E) · Δ` when species_corr is not available.

### `_build_cef_incoherent` — LTHR=2

Stores SB (bound cross section with optional redistribution factor) and Wp
(DW parameter of the principal scatterer, from `dwpix_out`).

### `_build_mef_elastic` — LTHR=3

Combines coherent Bragg edges (with per-species DW, same as CEF) and incoherent
elastic (SB = σ_inc of principal, Wp = principal's DW) in one section.

---

## 8. Validation

Tested against NCrystal reference ENDF files for β-SiC (F-43m, a=4.348 Å,
4 Si + 4 C atoms per unit cell) at 293.15 K:

| Case | LTHR | Metric | Our Value | Reference | Difference |
|------|------|--------|-----------|-----------|------------|
| C in SiC (CEF) | 1 | Final cumulative S | 1.313574 | 1.313143 | 0.033% |
| Si in SiC (CEF) | 2 | SB [barns] | 0.005 | 0.005 | exact |
| Si in SiC (CEF) | 2 | Wp [1/eV] | 1.265220 | 1.265463 | 0.019% |
| C in SiC (MEF) | 3 | Final cumulative S | 0.656787 | 0.656571 | 0.033% |
| C in SiC (MEF) | 3 | Wp [1/eV] | 1.541939 | 1.542509 | 0.037% |
| Si in SiC (MEF) | 3 | Wp [1/eV] | 1.265220 | 1.265463 | 0.019% |

The MEF cumulative S (0.656787) is exactly half the CEF DC-atom value
(1.313574 / 2), confirming correct per-atom normalization.

Mono-atomic materials (Al FCC, Fe BCC, graphite hexagonal) also verified
with iel=10 inputs against standard iel=1-6 outputs.

---

## 9. Design Decisions

| Decision | Rationale |
|---|---|
| DW not stored in bragg_data | Temperature-independent Bragg edges allow reuse across temperatures. DW applied at output time. |
| Per-species DW inside structure factor sum | Matches NCrystal exactly. A single effective DW gives ~3-6% error for polyatomic materials. |
| Species correlation matrix D_{st} | Avoids recomputing Bragg edges for per-species DW. Returned alongside bragg_data when `per_species=True`. |
| mult=2 per enumerated (h,k,l) | Counts Friedel pair; additional symmetry emerges from grouping by (d, F²). |
| Conservative h/k/l bounds | `ceil(a/dcutoff)+1` is always safe; slight over-enumeration is fast with early-exit on |k|². |
| Self-contained in leapr_generalized.py | No external module dependencies. Bragg-edge calculator, data structures, and builders all inlined. |
| Legacy coher() preserved | Backward compatibility with iel=1-6 hardcoded materials. |
