# Generalised Coherent Elastic Cross-Section: Algorithm and Integration Guide

This document explains the physics, the algorithm, and all code required to replace
a hardcoded Bragg-edge calculator (supporting only a fixed list of lattice types) with
a general implementation that works for any crystal structure.

The algorithm is derived from the NCrystal open-source neutron scattering library
(T. Kittelmann et al., *Computer Physics Communications* **267** (2021) 108082).
Specifically, the relevant NCrystal source files are:

| NCrystal file | Role |
|---|---|
| `NCLatticeUtils.cc` | Reciprocal lattice matrix construction |
| `NCFillHKL.cc` | HKL enumeration and structure factor |
| `NCPowderBragg.cc` | Cross-section assembly |
| `NCDefs.hh` | Physical constants |

---

## 1. Physics Background

### 1.1 Coherent elastic cross-section for a powder

For a polycrystalline material (powder), the coherent elastic neutron cross-section
per atom is (Squires, *Introduction to the Theory of Thermal Neutron Scattering*):

```
σ_coh_el(E) = (λ² / 4 V_atom) · Σ_{d_hkl ≥ λ/2}  mult_hkl · |F_hkl|² · d_hkl
```

where:

| Symbol | Meaning |
|---|---|
| `E` | Neutron kinetic energy [eV] |
| `λ = sqrt(WL2EKIN / E)` | Neutron wavelength [Å] |
| `WL2EKIN = 0.081804209605330899 eV·Å²` | `ℏ²/(2 m_n)` |
| `V_atom = V_cell / N_atoms` | Volume per atom [Å³] |
| `d_hkl` | d-spacing of plane set (h,k,l) [Å] |
| `mult_hkl` | Crystallographic multiplicity of (h,k,l) |
| `F_hkl` | Structure factor [√barn] |

The condition `d_hkl ≥ λ/2` means the Bragg condition can be satisfied: each new
plane set enters as E increases past the threshold:

```
E_threshold(d) = WL2EKIN / (4 d²)
```

Substituting `λ² = WL2EKIN / E` gives a convenient form with a running numerator:

```
σ(E) = Σ_j [barn·eV] / E [eV] = fdm_cumul / E
```

where each plane set contributes:

```
Δ(fdm_cumul) = d · |F|² · mult · 0.5 · WL2EKIN / (V_cell · N_atoms)   [barn·eV]
```

### 1.2 Structure factor

```
F(h,k,l) = Σ_{species s}  b_s [√barn] · Σ_{j ∈ s}  exp(i 2π (h·x_j + k·y_j + l·z_j))
```

- `b_s` = coherent scattering length [√barn].  To convert: `b [√barn] = b_coh_fm / 10`
  since `1 barn = 100 fm²`.
- `(x_j, y_j, z_j)` = fractional coordinates in the unit cell.
- `|F|²` has units [barn].

### 1.3 Debye-Waller (how it is handled here)

The Debye-Waller factor `exp(-2W)` where `W = ½ k² · MSD` attenuates the
structure factor at high momentum transfer.  The present algorithm stores
**temperature-independent** per-plane contributions (DW = 1, same as NJOY's
internal `coher` subroutine with `wint = 0`).  The output stage applies an
average DW via `dwpix` (computed from the phonon spectrum) as it writes to ENDF,
exactly as in the original LEAPR code.

If full per-species, per-temperature DW is needed in the future, each
`AtomSite` already carries a `b_coh_fm` field; a `msd` field and a
`Debye-Waller(T)` property can be added to `AtomSite` without changing any
other interface.

---

## 2. Output Data Format Contract

The function produces a 2-column NumPy array `bragg_data`:

```
bragg_data[:, 0]  = E_threshold [eV]     — ascending
bragg_data[:, 1]  = σ_contribution [barn·eV]  — per plane-group, NO DW
```

The ENDF writer accumulates these at output time:

```python
# for temperature itemp with DW parameter w = dwpix[itemp]:
cumulative = 0.0
for j in range(nbe):
    cumulative += exp(-4 * w * bragg_data[j, 0]) * bragg_data[j, 1]
    # write pair: (bragg_data[j, 0], cumulative)
```

The cumulative value [barn·eV] divided by energy [eV] gives the cross-section [barn].

The last row is a sentinel: `bragg_data[-1, 0] == emax` with the same
σ as the previous row, matching LEAPR convention.

---

## 3. New Data Structures

These must be defined **before** the input-parameters class so that
`crystal_structure: Optional[CrystalStructure] = None` can be used as a field.

```python
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class AtomSite:
    """One distinct atom species in the unit cell.

    Args:
        b_coh_fm:  Coherent scattering length [fm].
                   NIST 2018 values: C=6.646, Be=7.79, O=5.803, Al=3.449,
                                     Pb=9.405, Fe=9.45, Si=4.1491, N=9.36, …
        positions: List of (x, y, z) fractional coordinates for each atom
                   of this species within the unit cell.
    """
    b_coh_fm: float
    positions: List[Tuple[float, float, float]]

    @property
    def b_coh_sqrtbarn(self) -> float:
        """Scattering length in √barn  (1 barn = 100 fm²)."""
        return self.b_coh_fm / 10.0


@dataclass
class CrystalStructure:
    """Full crystal structure description for Bragg-edge calculation.

    Args:
        a, b, c:              Lattice parameters [Å].
        alpha, beta, gamma:   Lattice angles [degrees].
                              Convention: alpha between b and c,
                                          beta  between a and c,
                                          gamma between a and b.
        sites:                One AtomSite per distinct species.
    """
    a: float
    b: float
    c: float
    alpha: float
    beta:  float
    gamma: float
    sites: List[AtomSite]

    @property
    def n_atoms(self) -> int:
        """Total atoms per unit cell."""
        return sum(len(s.positions) for s in self.sites)

    @property
    def volume(self) -> float:
        """Unit-cell volume [Å³]."""
        ca = np.cos(np.radians(self.alpha))
        cb = np.cos(np.radians(self.beta))
        cg = np.cos(np.radians(self.gamma))
        return (self.a * self.b * self.c *
                np.sqrt(max(0.0, 1.0 - ca**2 - cb**2 - cg**2 + 2.0*ca*cb*cg)))
```

---

## 4. Physical Constant

```python
# WL2EKIN = ℏ²/(2 m_n)  [eV·Å²]
# Neutron kinetic energy: E = WL2EKIN / λ²  (λ in Å, E in eV)
# Source: NCrystal NCDefs.hh
WL2EKIN = 0.081804209605330899
```

---

## 5. Reciprocal Lattice Matrix

`get_reciprocal_lattice_matrix` maps integer Miller indices to Cartesian k-vectors:

```
k_vec [Å⁻¹] = G [3×3, Å⁻¹] @ [h, k, l]
d-spacing: d = 2π / |k_vec|
```

Algorithm mirrors `NCLatticeUtils.cc::getReciprocalLatticeRot`.
Special analytic cases are handled before the general `G = 2π L⁻¹` path.

```python
def get_reciprocal_lattice_matrix(a: float, b: float, c: float,
                                   alpha_deg: float, beta_deg: float,
                                   gamma_deg: float) -> np.ndarray:
    """
    Returns 3×3 numpy array G [Å⁻¹] such that k_vec = G @ [h, k, l].
    """
    tol  = 1e-10
    k2pi = 2.0 * np.pi
    alpha = np.radians(alpha_deg)
    beta  = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    a90  = abs(alpha - np.pi / 2) < tol
    b90  = abs(beta  - np.pi / 2) < tol
    g90  = abs(gamma - np.pi / 2) < tol
    g120 = abs(gamma - 2 * np.pi / 3) < tol

    if a90 and b90 and g90:
        # Cubic / tetragonal / orthorhombic
        return np.diag([k2pi / a, k2pi / b, k2pi / c])

    if a90 and b90 and g120:
        # Hexagonal
        sq3 = np.sqrt(3.0)
        return np.array([
            [k2pi / a,           0.0,                    0.0   ],
            [k2pi / (a * sq3),   2.0 * k2pi / (b * sq3), 0.0  ],
            [0.0,                0.0,                    k2pi/c],
        ])

    if a90 and g90:
        # Monoclinic (β ≠ 90°)
        sb   = np.sin(beta)
        cotb = np.cos(beta) / sb
        return np.array([
            [k2pi / a,          0.0,          0.0          ],
            [0.0,               k2pi / b,     0.0          ],
            [-cotb * k2pi / a,  0.0,          k2pi/(c*sb)  ],
        ])

    # General triclinic:  G = 2π · L⁻¹
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sb, sg     = np.sin(beta),  np.sin(gamma)
    m57 = c * (ca - cb * cg) / sg
    m88 = c * np.sqrt(max(0.0, sb**2 - ((ca - cb * cg) / sg)**2))
    L = np.array([
        [a,    b * cg,  c * cb],
        [0.0,  b * sg,  m57   ],
        [0.0,  0.0,     m88   ],
    ])
    return k2pi * np.linalg.inv(L)
```

---

## 6. Core Algorithm: `compute_bragg_edges_general`

This function is the direct replacement for the old hardcoded Bragg-edge calculator.
It is self-contained: it only requires `WL2EKIN`,
`get_reciprocal_lattice_matrix`, and `CrystalStructure`/`AtomSite`.

```python
def compute_bragg_edges_general(
    crystal: CrystalStructure,
    emax:       float,
    dcutoff:    float = 0.5,
    fsquarecut: float = 1e-5,
    merge_tol:  float = 1e-4,
) -> Tuple[np.ndarray, int]:
    """
    Compute Bragg-edge data for an arbitrary crystal structure.

    Based on NCrystal (NCFillHKL.cc + NCPowderBragg.cc).

    Parameters
    ----------
    crystal     : CrystalStructure
    emax        : Maximum Bragg-edge energy [eV]
    dcutoff     : Minimum d-spacing to consider [Å]  (default 0.5 Å)
    fsquarecut  : Minimum |F|² to keep a plane [barn]  (default 1e-5)
    merge_tol   : Relative tolerance for grouping planes with the same d

    Returns
    -------
    bragg_data  : ndarray, shape (N, 2)
                  col 0 = E_threshold [eV],  col 1 = σ_contribution [barn·eV]
                  Sorted ascending in energy. Last row is Emax sentinel.
    nbe         : int  — number of rows in bragg_data
    """
    V = crystal.volume   # [Å³]
    N = crystal.n_atoms  # atoms per unit cell

    # Cross-section prefactor  [eV/Å]
    # d[Å] · F²[barn] · mult · xsectfact  =  σ_plane [barn·eV]
    xsectfact = 0.5 * WL2EKIN / (V * N)

    # Reciprocal lattice matrix  k_vec = G @ [h, k, l]
    G = get_reciprocal_lattice_matrix(
        crystal.a, crystal.b, crystal.c,
        crystal.alpha, crystal.beta, crystal.gamma,
    )

    # |k|² upper bound from dcutoff  (d ≥ dcutoff  ⟺  |k| ≤ 2π/dcutoff)
    ksq_max = (2.0 * np.pi / dcutoff) ** 2

    # Conservative Miller-index search bounds  (a/dcutoff gives the rough maximum)
    h_max = max(1, int(np.ceil(crystal.a / dcutoff)) + 1)
    k_max = max(1, int(np.ceil(crystal.b / dcutoff)) + 1)
    l_max = max(1, int(np.ceil(crystal.c / dcutoff)) + 1)

    # Precompute scattering lengths and position arrays
    sites_data = [
        (s.b_coh_sqrtbarn, np.asarray(s.positions, dtype=float))
        for s in crystal.sites
    ]

    # ------------------------------------------------------------------ #
    # HKL enumeration (NCrystal convention):                               #
    #   h ≥ 0                                                              #
    #   h = 0  →  k ≥ 0                                                   #
    #   h = k = 0  →  l > 0                                               #
    # Each (h,k,l) represents the Friedel pair {(h,k,l), (-h,-k,-l)},    #
    # so multiplicity starts at 2.                                          #
    # ------------------------------------------------------------------ #
    plane_list = []   # entries: [d, F², mult=2]

    for h in range(0, h_max + 1):
        k_lo = 0 if h == 0 else -k_max
        for k in range(k_lo, k_max + 1):
            l_lo = 1 if (h == 0 and k == 0) else -l_max
            for l in range(l_lo, l_max + 1):

                k_vec = G @ np.array([h, k, l], dtype=float)
                ksq   = float(np.dot(k_vec, k_vec))

                if ksq < 1e-30 or ksq > ksq_max:
                    continue

                d     = 2.0 * np.pi / np.sqrt(ksq)   # [Å]
                E_thr = WL2EKIN / (4.0 * d * d)       # [eV]
                if E_thr > emax:
                    continue

                # Structure factor
                # F(h,k,l) = Σ_species b_s · Σ_j exp(i 2π (h xj + k yj + l zj))
                real_part = 0.0
                imag_part = 0.0
                for b_s, pos in sites_data:
                    phase = 2.0 * np.pi * (
                        h * pos[:, 0] + k * pos[:, 1] + l * pos[:, 2]
                    )
                    real_part += b_s * np.sum(np.cos(phase))
                    imag_part += b_s * np.sum(np.sin(phase))

                F2 = real_part**2 + imag_part**2   # [barn]
                if F2 < fsquarecut:
                    continue   # systematic absence or negligible

                plane_list.append([d, F2, 2])

    if not plane_list:
        return np.array([]).reshape(0, 2), 0

    # ------------------------------------------------------------------ #
    # Group planes that share the same d AND F²                            #
    # This naturally accumulates multiplicity for higher-symmetry systems: #
    #   e.g. FCC (100),(010),(001) all have the same d and F² → mult = 6  #
    # ------------------------------------------------------------------ #
    plane_list.sort(key=lambda x: -x[0])   # descending d = ascending E

    groups = []   # [d_rep, F2_rep, mult_total]
    for d, F2, mult in plane_list:
        merged = False
        for grp in groups:
            d_g, F2_g = grp[0], grp[1]
            if (abs(d - d_g) / d_g < merge_tol and
                    abs(F2 - F2_g) / max(F2_g, 1e-30) < merge_tol):
                grp[2] += mult
                merged = True
                break
        if not merged:
            groups.append([d, F2, mult])

    # Build output pairs (E_threshold, σ_plane [barn·eV])
    pairs = []
    for d, F2, mult in groups:
        E_thr = WL2EKIN / (4.0 * d * d)
        sigma = d * F2 * mult * xsectfact
        pairs.append([E_thr, sigma])

    pairs.sort(key=lambda p: p[0])

    # Merge pairs with nearly identical energies
    TOLER = 1e-6
    combined = []
    for E, sig in pairs:
        if combined and (E - combined[-1][0]) < TOLER:
            combined[-1][1] += sig
        else:
            combined.append([E, sig])

    bragg_data = np.array(combined, dtype=float)
    nbe = bragg_data.shape[0]

    # Append Emax sentinel (LEAPR/NJOY convention)
    if nbe > 0 and bragg_data[-1, 0] < emax:
        bragg_data = np.vstack([bragg_data, [emax, bragg_data[-1, 1]]])
        nbe += 1

    return bragg_data, nbe
```

---

## 7. Convenience Factory: `crystal_from_lat`

Provides backward compatibility with LEAPR's `lat = 1..6` convention.
Returns a `CrystalStructure` for each hardcoded lattice type so that
existing workflows that set `lat` can be migrated transparently.

```python
def crystal_from_lat(lat: int) -> CrystalStructure:
    """
    Build a CrystalStructure for one of the six classic LEAPR lattice types.
    Scattering lengths are NIST 2018 values.

    lat=1  Graphite (hexagonal P6₃/mmc)
    lat=2  Beryllium (hexagonal P6₃/mmc)
    lat=3  Beryllium oxide (wurtzite P6₃mc)
    lat=4  Aluminium (FCC Fm-3m)
    lat=5  Lead (FCC Fm-3m)
    lat=6  Iron (BCC Im-3m)
    """
    if lat == 1:
        return CrystalStructure(
            a=2.4573, b=2.4573, c=6.700,
            alpha=90.0, beta=90.0, gamma=120.0,
            sites=[AtomSite(
                b_coh_fm=6.646,   # C
                positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.5),
                           (1/3, 2/3, 0.0), (2/3, 1/3, 0.5)],
            )],
        )
    elif lat == 2:
        return CrystalStructure(
            a=2.2856, b=2.2856, c=3.5832,
            alpha=90.0, beta=90.0, gamma=120.0,
            sites=[AtomSite(
                b_coh_fm=7.79,    # Be
                positions=[(0.0, 0.0, 0.0), (1/3, 2/3, 0.5)],
            )],
        )
    elif lat == 3:
        return CrystalStructure(
            a=2.695, b=2.695, c=4.39,
            alpha=90.0, beta=90.0, gamma=120.0,
            sites=[
                AtomSite(b_coh_fm=7.79,
                         positions=[(0.0, 0.0, 0.0), (1/3, 2/3, 0.5)]),      # Be
                AtomSite(b_coh_fm=5.803,
                         positions=[(0.0, 0.0, 0.375), (1/3, 2/3, 0.875)]),  # O
            ],
        )
    elif lat == 4:
        return CrystalStructure(
            a=4.04, b=4.04, c=4.04,
            alpha=90.0, beta=90.0, gamma=90.0,
            sites=[AtomSite(
                b_coh_fm=3.449,   # Al
                positions=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.5),
                           (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
            )],
        )
    elif lat == 5:
        return CrystalStructure(
            a=4.94, b=4.94, c=4.94,
            alpha=90.0, beta=90.0, gamma=90.0,
            sites=[AtomSite(
                b_coh_fm=9.405,   # Pb
                positions=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.5),
                           (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
            )],
        )
    elif lat == 6:
        return CrystalStructure(
            a=2.86, b=2.86, c=2.86,
            alpha=90.0, beta=90.0, gamma=90.0,
            sites=[AtomSite(
                b_coh_fm=9.45,    # Fe
                positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
            )],
        )
    else:
        raise ValueError(f"Unknown lat={lat}. Use 1..6 or supply CrystalStructure directly.")
```

---

## 8. Integration: What Changes in the Host Codebase

### 8.1 Input parameters class

Add one optional field to whatever class holds run parameters:

```python
crystal_structure: Optional[CrystalStructure] = None
# When set, this takes precedence over any legacy `lat` integer flag.
```

### 8.2 Bragg-edge dispatcher

Replace the body of the function that decides *which* Bragg-edge calculator
to call with this routing logic:

```python
def process_coherent_elastic(self, emax: float) -> Optional[Tuple[np.ndarray, int]]:
    params = self.params   # or however you access the run parameters

    crystal = getattr(params, 'crystal_structure', None)
    if crystal is not None:
        bragg_data, nbe = compute_bragg_edges_general(crystal, emax)
    elif getattr(params, 'lat', 0) != 0:
        bragg_data, nbe = compute_bragg_edges_legacy(params.lat, params.natom, emax)
    else:
        return None   # coherent elastic not requested

    # --- whatever post-processing (thinning, DW, output) was done before ---
    bragg_data, nbe = thin_bragg_edges(bragg_data, ...)
    return bragg_data, nbe
```

### 8.3 ENDF / output writer (no change required)

The `bragg_data` array produced by `compute_bragg_edges_general` has exactly
the same format as the old hardcoded function — per-plane contributions in
`[barn·eV]` without Debye-Waller.  Any downstream accumulation loop of the form:

```python
cumulative += exp(-4 * w * E) * sigma
```

continues to work unchanged.

---

## 9. Usage Examples

### Example A — drop-in replacement for lat=4 (Aluminium)

```python
# Old way:
bragg_data, nbe = compute_bragg_edges(lat=4, natom=4, emax=5.0)

# New way — identical crystal, general algorithm:
crystal = crystal_from_lat(4)
bragg_data, nbe = compute_bragg_edges_general(crystal, emax=5.0)
```

### Example B — arbitrary crystal (e.g. silicon)

```python
silicon = CrystalStructure(
    a=5.4307, b=5.4307, c=5.4307,
    alpha=90.0, beta=90.0, gamma=90.0,
    sites=[AtomSite(
        b_coh_fm=4.1491,   # Si (NIST 2018)
        positions=[
            (0.00, 0.00, 0.00), (0.50, 0.50, 0.00),
            (0.50, 0.00, 0.50), (0.00, 0.50, 0.50),
            (0.25, 0.25, 0.25), (0.75, 0.75, 0.25),
            (0.75, 0.25, 0.75), (0.25, 0.75, 0.75),
        ],
    )],
)
bragg_data, nbe = compute_bragg_edges_general(silicon, emax=5.0)
```

### Example C — multi-species crystal (e.g. SiC wurtzite)

```python
sic = CrystalStructure(
    a=3.073, b=3.073, c=5.053,
    alpha=90.0, beta=90.0, gamma=120.0,
    sites=[
        AtomSite(b_coh_fm=4.1491,   # Si
                 positions=[(0.0, 0.0, 0.0), (1/3, 2/3, 0.5)]),
        AtomSite(b_coh_fm=6.646,    # C
                 positions=[(0.0, 0.0, 0.375), (1/3, 2/3, 0.875)]),
    ],
)
bragg_data, nbe = compute_bragg_edges_general(sic, emax=5.0)
```

---

## 10. Key Design Decisions and Limitations

| Decision | Rationale |
|---|---|
| DW not in F² | Matches NJOY `coher` (wint=0). DW applied at output via dwpix. |
| mult=2 per enumerated (h,k,l) | Counts Friedel pair; extra symmetry emerges from grouping by (d, F²). |
| Conservative h/k/l bounds | `ceil(a/dcutoff)+1` is always safe; slight over-enumeration is fast with early-exit. |
| Per-plane σ [barn·eV] | Same unit and format as legacy function; ENDF writer needs no changes. |
| Last row = Emax sentinel | Required by LEAPR convention for the output thinning algorithm. |

**Limitations to be aware of:**
- For multi-species materials, the single average DW from `dwpix` is an
  approximation.  If per-species MSD data are available (e.g. from `calc_msd.py`),
  the DW can be incorporated directly into F² by multiplying each `b_s` by
  `exp(-0.5 * k² * msd_s)` inside the structure factor sum; this would require
  a separate `bragg_data` per temperature and a corresponding change to the writer.
- The multiplicity counting from the (d, F²) grouping is exact for Laue classes
  that are already covered by the chosen HKL half-space.  For space groups with
  true absences (glide planes, screw axes), the structure factor goes to zero and
  the `fsquarecut` filter removes those planes automatically.
