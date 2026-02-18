"""
coherent_elastic_general.py
============================
General-purpose coherent elastic (Bragg-edge) cross-section calculator for
polycrystalline materials.

This module is self-contained: it only depends on NumPy.  It can be dropped
into any neutron transport or thermal-scattering code that needs to compute
coherent elastic cross-sections for arbitrary crystal structures.

Algorithm
---------
Based on the NCrystal open-source library:
  T. Kittelmann et al., Computer Physics Communications 267 (2021) 108082
  Source files: NCLatticeUtils.cc, NCFillHKL.cc, NCPowderBragg.cc, NCDefs.hh

Physics
-------
The coherent elastic cross-section per atom for a powder is:

    σ(E) = (λ²/4 V_atom) · Σ_{d ≥ λ/2}  mult · |F(hkl)|² · d

where λ = sqrt(WL2EKIN/E), so conveniently:

    σ(E) = (1/E) · Σ_{E_thr(d) ≤ E}  d · |F|² · mult · WL2EKIN/(2 V_cell N)

Each plane set contributes a temperature-independent numerator chunk
    Δ = d · |F|² · mult · 0.5 · WL2EKIN / (V_cell · N_atoms)   [barn·eV]

and the cross-section is σ(E) = Σ Δ / E.

Output format
-------------
The function returns a 2-column NumPy array `bragg_data`:
    col 0 = E_threshold [eV]      (ascending)
    col 1 = Δ [barn·eV]           (per plane-group, Debye-Waller NOT included)

An ENDF/output writer accumulates the Δ values with an optional DW weight:
    cumulative += exp(-4 w E) · Δ
and writes (E, cumulative) pairs.  The last row is an Emax sentinel.

Usage
-----
    from coherent_elastic_general import (
        AtomSite, CrystalStructure, WL2EKIN,
        get_reciprocal_lattice_matrix,
        compute_bragg_edges_general,
        crystal_from_lat,
    )

    # Arbitrary crystal
    al = CrystalStructure(
        a=4.04, b=4.04, c=4.04, alpha=90, beta=90, gamma=90,
        sites=[AtomSite(b_coh_fm=3.449,
                        positions=[(0,0,0),(0,.5,.5),(.5,0,.5),(.5,.5,0)])],
    )
    bragg_data, nbe = compute_bragg_edges_general(al, emax=5.0)

    # Or use the convenience factory for LEAPR's lat=1..6:
    bragg_data, nbe = compute_bragg_edges_general(crystal_from_lat(4), emax=5.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Physical constant
# ---------------------------------------------------------------------------

# WL2EKIN = ℏ²/(2 m_n)  [eV·Å²]
# Neutron kinetic energy: E [eV] = WL2EKIN / λ² [Å²]
# Source: NCrystal NCDefs.hh  (CODATA 2018)
WL2EKIN: float = 0.081804209605330899


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AtomSite:
    """One distinct atom species occupying one or more sites in the unit cell.

    Parameters
    ----------
    b_coh_fm : float
        Coherent scattering length [fm].
        Selected NIST 2018 values:
            H   = -3.739,  D   =  6.671,  C   =  6.646,
            N   =  9.36,   O   =  5.803,  Si  =  4.1491,
            Al  =  3.449,  Fe  =  9.45,   Pb  =  9.405,
            Be  =  7.79,   Zr  =  7.16,   U   =  8.417
    positions : list of (x, y, z)
        Fractional coordinates of each atom of this species in the unit cell.
        Each entry is a 3-tuple of floats in [0, 1).
    """
    b_coh_fm: float
    positions: List[Tuple[float, float, float]]

    @property
    def b_coh_sqrtbarn(self) -> float:
        """Coherent scattering length in √barn  (1 barn = 100 fm²)."""
        return self.b_coh_fm / 10.0


@dataclass
class CrystalStructure:
    """Full crystal structure description.

    Parameters
    ----------
    a, b, c : float
        Lattice parameters [Å].
    alpha, beta, gamma : float
        Lattice angles [degrees].
        Convention (standard crystallographic):
            alpha = angle between b and c axes
            beta  = angle between a and c axes
            gamma = angle between a and b axes
    sites : list of AtomSite
        One entry per distinct atom species.  Multiple species are allowed
        (e.g. BeO has two sites: one for Be, one for O).
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
        """Total atoms per unit cell (sum over all sites)."""
        return sum(len(s.positions) for s in self.sites)

    @property
    def volume(self) -> float:
        """Unit-cell volume [Å³]."""
        ca = np.cos(np.radians(self.alpha))
        cb = np.cos(np.radians(self.beta))
        cg = np.cos(np.radians(self.gamma))
        # Standard formula: V = a b c sqrt(1 - cos²α - cos²β - cos²γ + 2 cosα cosβ cosγ)
        return (self.a * self.b * self.c *
                np.sqrt(max(0.0, 1.0 - ca**2 - cb**2 - cg**2 + 2.0*ca*cb*cg)))


# ---------------------------------------------------------------------------
# Reciprocal lattice matrix
# ---------------------------------------------------------------------------

def get_reciprocal_lattice_matrix(a: float, b: float, c: float,
                                   alpha_deg: float, beta_deg: float,
                                   gamma_deg: float) -> np.ndarray:
    """Compute the 3×3 reciprocal lattice matrix G.

    G maps integer Miller indices to Cartesian k-vectors:
        k_vec [Å⁻¹] = G @ [h, k, l]
        d-spacing [Å] = 2π / |k_vec|

    Algorithm mirrors NCrystal NCLatticeUtils.cc (getReciprocalLatticeRot).
    Analytic special cases are used for the common symmetries; the general
    case falls back to G = 2π · L⁻¹.

    Parameters
    ----------
    a, b, c : float
        Lattice parameters [Å].
    alpha_deg, beta_deg, gamma_deg : float
        Lattice angles [degrees].

    Returns
    -------
    G : ndarray, shape (3, 3), dtype float
        Units: Å⁻¹.
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
        # Cubic / tetragonal / orthorhombic  (all angles = 90°)
        return np.diag([k2pi / a, k2pi / b, k2pi / c])

    if a90 and b90 and g120:
        # Hexagonal  (α = β = 90°, γ = 120°)
        sq3 = np.sqrt(3.0)
        return np.array([
            [k2pi / a,             0.0,                    0.0    ],
            [k2pi / (a * sq3),  2.0 * k2pi / (b * sq3),   0.0    ],
            [0.0,               0.0,                    k2pi / c  ],
        ])

    if a90 and g90:
        # Monoclinic  (α = γ = 90°, β ≠ 90°)
        sb   = np.sin(beta)
        cotb = np.cos(beta) / sb
        return np.array([
            [k2pi / a,            0.0,        0.0          ],
            [0.0,                 k2pi / b,   0.0          ],
            [-cotb * k2pi / a,    0.0,        k2pi/(c*sb)  ],
        ])

    # General triclinic:  G = 2π · L⁻¹
    # L is the lower-triangular matrix whose columns are the real-space basis vectors.
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


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def compute_bragg_edges_general(
    crystal:     CrystalStructure,
    emax:        float,
    dcutoff:     float = 0.5,
    fsquarecut:  float = 1e-5,
    merge_tol:   float = 1e-4,
) -> Tuple[np.ndarray, int]:
    """Compute Bragg-edge cross-section data for any polycrystalline material.

    This is a general replacement for hardcoded single-lattice Bragg-edge
    calculators.  It correctly handles:
    - Any crystal symmetry (triclinic through cubic)
    - Multiple atom species per unit cell (e.g. BeO, SiC, UO₂)
    - Systematic absences (structure factor vanishes → filtered by fsquarecut)
    - Crystallographic multiplicity (planes with equal d and F² are merged)

    The algorithm follows NCrystal (NCFillHKL.cc + NCPowderBragg.cc):
        T. Kittelmann et al., CPC 267 (2021) 108082

    Parameters
    ----------
    crystal : CrystalStructure
        Lattice parameters, angles, and atom sites.
    emax : float
        Maximum Bragg-edge energy to include [eV].  Edges above this are
        ignored (they correspond to planes with d so small that the Bragg
        condition requires E > emax).
    dcutoff : float, optional
        Minimum d-spacing [Å] to consider.  Default 0.5 Å (NCrystal default).
        Planes with d < dcutoff are excluded — they contribute negligibly at
        practical neutron energies and would require very large Miller indices.
    fsquarecut : float, optional
        Minimum structure factor |F|² [barn] to keep a plane.  Default 1e-5.
        This filters systematic absences (|F|² = 0) and near-zero planes.
    merge_tol : float, optional
        Relative tolerance for grouping planes that share the same d-spacing
        and F² into one entry (accumulating multiplicity).  Default 1e-4.

    Returns
    -------
    bragg_data : ndarray, shape (N, 2)
        Column 0: Bragg-edge energy E_threshold [eV], sorted ascending.
        Column 1: Per-plane-group cross-section contribution [barn·eV].
                  Debye-Waller is NOT included (temperature-independent).
        The last row is an Emax sentinel: bragg_data[-1, 0] == emax,
        bragg_data[-1, 1] == bragg_data[-2, 1].
    nbe : int
        Number of rows in bragg_data.

    Notes
    -----
    Debye-Waller exclusion:
        The stored values are temperature-independent (DW = 1), matching
        NJOY LEAPR's internal ``coher`` subroutine (wint = 0).  The DW
        factor is applied at output time when accumulating the cumulative
        cross-section for the ENDF file.

    Output accumulation (ENDF writer side):
        cumulative = 0
        for j in range(nbe):
            E     = bragg_data[j, 0]
            delta = bragg_data[j, 1]
            cumulative += exp(-4 * w * E) * delta   # w = DW parameter
            write_pair(E, cumulative)               # (E [eV], cumul [barn·eV])
        # cumulative / E  gives the physical cross-section [barn] at energy E
    """
    V = crystal.volume   # unit-cell volume [Å³]
    N = crystal.n_atoms  # atoms per unit cell

    if V <= 0.0:
        raise ValueError(f"Non-positive unit-cell volume {V:.6g} Å³. "
                         "Check lattice parameters.")
    if N == 0:
        raise ValueError("Crystal has no atom sites.")

    # ------------------------------------------------------------------
    # Cross-section prefactor
    # xsectfact [eV/Å]:  d[Å] · F²[barn] · mult · xsectfact = Δ [barn·eV]
    # Derived from the Squires powder formula:
    #   σ = λ²/(4 V_atom) · Σ mult·|F|²·d  =  Σ Δ / E
    # ------------------------------------------------------------------
    xsectfact = 0.5 * WL2EKIN / (V * N)

    # Reciprocal lattice matrix  (k_vec = G @ [h,k,l])
    G = get_reciprocal_lattice_matrix(
        crystal.a, crystal.b, crystal.c,
        crystal.alpha, crystal.beta, crystal.gamma,
    )

    # Upper bound on |k|²: planes with d < dcutoff have |k| > 2π/dcutoff
    ksq_max = (2.0 * np.pi / dcutoff) ** 2

    # Conservative Miller-index search bounds
    # The largest h needed is when G[0,0]*h ≈ 2π/dcutoff, i.e. h ≈ a/dcutoff.
    h_max = max(1, int(np.ceil(crystal.a / dcutoff)) + 1)
    k_max = max(1, int(np.ceil(crystal.b / dcutoff)) + 1)
    l_max = max(1, int(np.ceil(crystal.c / dcutoff)) + 1)

    # Precompute per-site scattering lengths and position arrays
    sites_data = [
        (s.b_coh_sqrtbarn, np.asarray(s.positions, dtype=float))
        for s in crystal.sites
    ]

    # ------------------------------------------------------------------
    # HKL enumeration — NCrystal convention
    #
    # We only enumerate the "positive half" of reciprocal space:
    #   h ≥ 0
    #   h = 0   →  k ≥ 0
    #   h = k = 0  →  l > 0
    #
    # Each such (h,k,l) represents the Friedel pair {(h,k,l), (-h,-k,-l)},
    # so its initial multiplicity is 2.  Additional symmetry-equivalent
    # planes (same d, same F²) are discovered in the grouping step below.
    # ------------------------------------------------------------------
    plane_list: List[List[float]] = []   # each entry: [d, F², mult=2]

    for h in range(0, h_max + 1):
        k_lo = 0 if h == 0 else -k_max
        for k in range(k_lo, k_max + 1):
            l_lo = 1 if (h == 0 and k == 0) else -l_max
            for l in range(l_lo, l_max + 1):

                k_vec = G @ np.array([h, k, l], dtype=float)
                ksq   = float(np.dot(k_vec, k_vec))

                if ksq < 1e-30 or ksq > ksq_max:
                    continue   # (0,0,0) or d < dcutoff

                d     = 2.0 * np.pi / np.sqrt(ksq)   # [Å]
                E_thr = WL2EKIN / (4.0 * d * d)       # [eV]  Bragg threshold
                if E_thr > emax:
                    continue   # plane only reachable above energy range

                # ------------------------------------------------------
                # Structure factor
                # F(hkl) = Σ_species b_s · Σ_j∈species exp(i 2π hkl·r_j)
                # |F|² = real² + imag²   [barn]
                # ------------------------------------------------------
                real_part = 0.0
                imag_part = 0.0
                for b_s, pos in sites_data:
                    phase = 2.0 * np.pi * (
                        h * pos[:, 0] + k * pos[:, 1] + l * pos[:, 2]
                    )
                    real_part += b_s * float(np.sum(np.cos(phase)))
                    imag_part += b_s * float(np.sum(np.sin(phase)))

                F2 = real_part**2 + imag_part**2   # [barn]

                if F2 < fsquarecut:
                    continue   # systematic absence or negligible reflection

                plane_list.append([d, F2, 2.0])   # mult=2: Friedel pair

    if not plane_list:
        return np.empty((0, 2), dtype=float), 0

    # ------------------------------------------------------------------
    # Group planes with the same d-spacing AND F²
    #
    # This step naturally discovers crystallographic multiplicity for any
    # symmetry.  Example: FCC (100), (010), (001) each appear with mult=2,
    # but all share the same d = a and F² = 16 b²; they merge into one
    # group with mult_total = 6.
    # ------------------------------------------------------------------
    plane_list.sort(key=lambda x: -x[0])   # descending d  ≡  ascending E_thr

    groups: List[List[float]] = []   # [d_rep, F2_rep, mult_total]
    for d, F2, mult in plane_list:
        merged = False
        for grp in groups:
            d_g, F2_g = grp[0], grp[1]
            if (abs(d   - d_g)  /      d_g         < merge_tol and
                abs(F2  - F2_g) / max(F2_g, 1e-30) < merge_tol):
                grp[2] += mult
                merged = True
                break
        if not merged:
            groups.append([d, F2, mult])

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    pairs: List[List[float]] = []
    for d, F2, mult in groups:
        E_thr = WL2EKIN / (4.0 * d * d)
        sigma = d * F2 * mult * xsectfact   # [barn·eV]
        pairs.append([E_thr, sigma])

    pairs.sort(key=lambda p: p[0])

    # Merge pairs that fell within absolute energy tolerance (numerical noise)
    TOLER = 1e-6   # [eV]
    combined: List[List[float]] = []
    for E, sig in pairs:
        if combined and (E - combined[-1][0]) < TOLER:
            combined[-1][1] += sig   # same energy bin
        else:
            combined.append([E, sig])

    bragg_data = np.array(combined, dtype=float)
    nbe = int(bragg_data.shape[0])

    # Emax sentinel — matches LEAPR/NJOY convention
    if nbe > 0 and bragg_data[-1, 0] < emax:
        bragg_data = np.vstack([bragg_data, [emax, bragg_data[-1, 1]]])
        nbe += 1

    return bragg_data, nbe


# ---------------------------------------------------------------------------
# Convenience factory for LEAPR lat=1..6 backward compatibility
# ---------------------------------------------------------------------------

def crystal_from_lat(lat: int) -> CrystalStructure:
    """Return a CrystalStructure for one of the six classic LEAPR lattice types.

    This allows existing workflows that use the LEAPR ``lat`` integer flag to
    migrate to the general algorithm without changing their crystal data.
    Scattering lengths are NIST 2018 values.

    Parameters
    ----------
    lat : int
        1 = Graphite  (hexagonal P6₃/mmc, 4 C  atoms/cell)
        2 = Beryllium (hexagonal P6₃/mmc, 2 Be atoms/cell)
        3 = BeO       (wurtzite  P6₃mc,   2 Be + 2 O atoms/cell)
        4 = Aluminium (FCC Fm-3m,          4 Al atoms/cell)
        5 = Lead      (FCC Fm-3m,          4 Pb atoms/cell)
        6 = Iron      (BCC Im-3m,          2 Fe atoms/cell)

    Returns
    -------
    CrystalStructure

    Raises
    ------
    ValueError : if lat is not in 1..6
    """
    if lat == 1:
        # Graphite — hexagonal P6₃/mmc  (AB stacking)
        # ABAB stacking: A layer at z=0, B layer at z=1/2
        # A sites: (0,0,0) and (2/3,1/3,0) — but NJOY uses (0,0,0)(1/3,2/3,0)
        # B sites: (0,0,1/2) and (2/3,1/3,1/2) — NJOY: (0,0,1/2)(2/3,1/3,1/2)
        return CrystalStructure(
            a=2.4573, b=2.4573, c=6.700,
            alpha=90.0, beta=90.0, gamma=120.0,
            sites=[AtomSite(
                b_coh_fm=6.646,   # C (NIST 2018)
                positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.5),
                           (1/3, 2/3, 0.0), (2/3, 1/3, 0.5)],
            )],
        )
    elif lat == 2:
        # Beryllium — hexagonal P6₃/mmc
        return CrystalStructure(
            a=2.2856, b=2.2856, c=3.5832,
            alpha=90.0, beta=90.0, gamma=120.0,
            sites=[AtomSite(
                b_coh_fm=7.79,    # Be (NIST 2018)
                positions=[(0.0, 0.0, 0.0), (1/3, 2/3, 0.5)],
            )],
        )
    elif lat == 3:
        # Beryllium oxide — wurtzite P6₃mc
        # Be at (0,0,0) and (1/3,2/3,1/2); O at (0,0,u) and (1/3,2/3,1/2+u), u≈0.375
        return CrystalStructure(
            a=2.695, b=2.695, c=4.39,
            alpha=90.0, beta=90.0, gamma=120.0,
            sites=[
                AtomSite(b_coh_fm=7.79,    # Be
                         positions=[(0.0, 0.0, 0.0), (1/3, 2/3, 0.5)]),
                AtomSite(b_coh_fm=5.803,   # O (NIST 2018)
                         positions=[(0.0, 0.0, 0.375), (1/3, 2/3, 0.875)]),
            ],
        )
    elif lat == 4:
        # Aluminium — FCC Fm-3m  (conventional cell, 4 atoms)
        return CrystalStructure(
            a=4.04, b=4.04, c=4.04,
            alpha=90.0, beta=90.0, gamma=90.0,
            sites=[AtomSite(
                b_coh_fm=3.449,   # Al (NIST 2018)
                positions=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.5),
                           (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
            )],
        )
    elif lat == 5:
        # Lead — FCC Fm-3m  (conventional cell, 4 atoms)
        return CrystalStructure(
            a=4.94, b=4.94, c=4.94,
            alpha=90.0, beta=90.0, gamma=90.0,
            sites=[AtomSite(
                b_coh_fm=9.405,   # Pb (NIST 2018)
                positions=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.5),
                           (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
            )],
        )
    elif lat == 6:
        # Iron — BCC Im-3m  (2 atoms per conventional cell)
        return CrystalStructure(
            a=2.86, b=2.86, c=2.86,
            alpha=90.0, beta=90.0, gamma=90.0,
            sites=[AtomSite(
                b_coh_fm=9.45,    # Fe (NIST 2018)
                positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
            )],
        )
    else:
        raise ValueError(
            f"Unknown lat={lat}. "
            "Valid values are 1 (graphite), 2 (Be), 3 (BeO), "
            "4 (Al), 5 (Pb), 6 (Fe). "
            "For any other material supply a CrystalStructure directly."
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== coherent_elastic_general.py self-test ===\n")

    # ---- 1. Graphite unit-cell volume -----------------------------------------
    gr = crystal_from_lat(1)
    expected_vol = gr.a**2 * gr.c * np.sqrt(3) / 2
    print(f"Graphite: N={gr.n_atoms}, V={gr.volume:.4f} Å³ (expected {expected_vol:.4f} Å³)")
    assert abs(gr.volume - expected_vol) < 1e-4

    # ---- 2. Reciprocal lattice for Al (cubic) ---------------------------------
    G = get_reciprocal_lattice_matrix(4.04, 4.04, 4.04, 90, 90, 90)
    d111 = 2 * np.pi / np.linalg.norm(G @ np.array([1, 1, 1]))
    d200 = 2 * np.pi / np.linalg.norm(G @ np.array([2, 0, 0]))
    print(f"Al  d(111)={d111:.4f} Å (exp {4.04/np.sqrt(3):.4f}), "
          f"d(200)={d200:.4f} Å (exp {4.04/2:.4f})")
    assert abs(d111 - 4.04 / np.sqrt(3)) < 1e-4
    assert abs(d200 - 4.04 / 2) < 1e-4

    # ---- 3. Graphite (001) systematic absence ---------------------------------
    b  = gr.sites[0].b_coh_sqrtbarn
    pos = np.asarray(gr.sites[0].positions)
    for h, k, l, should_vanish in [(0,0,1,True), (0,0,2,False), (1,0,0,False)]:
        phase = 2*np.pi*(h*pos[:,0] + k*pos[:,1] + l*pos[:,2])
        F2 = (b*np.sum(np.cos(phase)))**2 + (b*np.sum(np.sin(phase)))**2
        tag = "ABSENT" if should_vanish else "present"
        print(f"  Graphite F²({h}{k}{l}) = {F2:.4f} barn  ({tag})")
        if should_vanish:
            assert F2 < 1e-10, f"Expected absence for ({h}{k}{l})"

    # ---- 4. Full Bragg-edge calculation for Al --------------------------------
    al = crystal_from_lat(4)
    bd, nbe = compute_bragg_edges_general(al, emax=5.0)
    E_first = bd[0, 0]
    E_expected_111 = WL2EKIN / (4 * (al.a / np.sqrt(3))**2)
    print(f"\nAl: {nbe} edges,  first edge E={E_first:.5f} eV "
          f"(expected {E_expected_111:.5f} eV for 111)")
    assert abs(E_first - E_expected_111) < 1e-4
    assert bd[-1, 0] == 5.0,  "Last row should be Emax sentinel"
    print("  bd[-1] =", bd[-1], "  ← Emax sentinel ✓")

    # ---- 5. BeO two-species structure factor ----------------------------------
    beo = crystal_from_lat(3)
    bd_beo, nbe_beo = compute_bragg_edges_general(beo, emax=5.0)
    print(f"\nBeO: {nbe_beo} edges,  first 3:\n{bd_beo[:3]}")
    assert nbe_beo > 0

    print("\nAll self-tests passed.")
