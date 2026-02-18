#!/usr/bin/env python3
"""
Python translation of the NJOY2016 LEAPR module with generalized elastic
scattering for arbitrary crystal structures.

Calculates the thermal neutron scattering law S(alpha, beta) using the
phonon expansion method. Supports:
  - Continuous phonon frequency distributions
  - Free-gas and diffusion translational modes
  - Discrete oscillators
  - Cold hydrogen/deuterium (ortho/para)
  - Coherent elastic (Bragg edges) for graphite, Be, BeO, Al, Pb, Fe (iel=1-6)
  - Generalized coherent elastic for arbitrary crystals (iel=10)
      * CEF: Current ENDF Format (elastic_mode=1)
      * MEF: Mixed Elastic Format (elastic_mode=2, LTHR=3)
      * Per-species Debye-Waller factors from partial phonon spectra
      * Polyatomic unit cells with DC-atom selection (Ramic et al., NIM-A 1027, 2022)
  - Incoherent elastic
  - Mixed moderators (secondary scatterers)
  - Skold approximation for intermolecular coherence

Output is written in ENDF-6 MF7/MT4 format using endf-parserpy.

Usage:
    python leapr_generalized.py <input_file> <output_file>


Generalized Elastic Input Cards (iel=10)
=========================================

When iel=10 is specified on Card 5, additional input cards are read after
Card 6 (secondary scatterer control) to define the crystal structure and
elastic scattering treatment. These cards follow the formalism described in:

    K. Ramic, J. I. Damian Marquez, et al., "NJOY+NCrystal: An open-source
    tool for creating thermal neutron scattering libraries with mixed elastic
    support", NIM-A 1027 (2022) 166227.

Card 6b — Elastic mode and atom counts
---------------------------------------
    elastic_mode  nat  nspec  /

    elastic_mode  1 = CEF (Current ENDF Format)
                      - Dominant-channel (DC) atom gets LTHR=1 (coherent)
                      - Other atoms get LTHR=2 (incoherent) with redistribution
                      - For single-atom materials: σ_coh vs σ_inc determines channel
                  2 = MEF (Mixed Elastic Format)
                      - All atoms get LTHR=3 (both coherent + incoherent)
                      - Coherent part: per-atom Bragg edges
                      - Incoherent part: per-atom σ_inc and DW factor
    nat           Number of distinct atom types in the unit cell (>= 1)
    nspec         Number of partial phonon spectra to follow

Card 6c — Lattice parameters
-----------------------------
    a  b  c  alpha  beta  gamma  /

    a, b, c         Lattice constants [Angstrom]
    alpha, beta, gamma  Lattice angles [degrees]
                        (alpha=b^c, beta=a^c, gamma=a^b)

Card 6d — Atom types (repeated nat times)
------------------------------------------
  For each atom type i = 1, ..., nat:

    Z_i  A_i  awr_i  b_coh_i  sigma_inc_i  npos_i  /
    x1 y1 z1  x2 y2 z2  ...  (npos_i positions)  /

    Z_i           Atomic number
    A_i           Mass number
    awr_i         Atomic weight ratio (to neutron mass)
    b_coh_i       Coherent scattering length [fm]
    sigma_inc_i   Incoherent scattering cross section [barns]
    npos_i        Number of positions of this atom in the unit cell
    x y z         Fractional coordinates of each position

Card 6e — Partial phonon spectra (repeated nspec times)
--------------------------------------------------------
  For each spectrum s = 1, ..., nspec:

    Z_s  A_s  delta_s  ni_s  /
    rho(1) rho(2) ... rho(ni_s)  /

    Z_s, A_s      Identifies which atom type this spectrum belongs to
                  (matched to Card 6d entries by Z and A)
    delta_s       Energy grid spacing [eV]
    ni_s          Number of phonon spectrum values
    rho(j)        Phonon density of states on equidistant grid

  These spectra are used to compute per-species Debye-Waller factors.
  Each atom type should have a matching partial spectrum; unmatched atom
  types fall back to the principal scatterer's DW factor.


Example Input — C in SiC (CEF, polyatomic)
============================================

  leapr
  20
  'C in SiC-beta (generalized CEF iel=10)'/
  1 1 100 /                          --- Card 3: ntempr=1, iprint=1, nphon=100
  101 6012./                          --- Card 4: mat=101, za=6012 (C-12)
  11.907856 4.724629 1 10 0/         --- Card 5: awr, spr, npr=1, iel=10, ncold=0
  0/                                  --- Card 6: nss=0 (no secondary scatterer)
  1 2 2/                              --- Card 6b: CEF, 2 atom types, 2 spectra
  4.348 4.348 4.348 90. 90. 90./     --- Card 6c: cubic SiC, a=4.348 Angstrom
  14 28 27.844241 4.1491 0.004 4/    --- Card 6d[1]: Si-28, b_coh=4.15 fm, 4 pos
  0 0 0  0 0.5 0.5  0.5 0 0.5  0.5 0.5 0/
  6 12 11.907856 6.646 0.001 4/      --- Card 6d[2]: C-12, b_coh=6.65 fm, 4 pos
  0.25 0.25 0.25  0.25 0.75 0.75  0.75 0.25 0.75  0.75 0.75 0.25/
  14 28 3.899867e-04 301/            --- Card 6e[1]: Si partial phonon DOS
  0.0 2.784318e-05 ... /
  6 12 3.899867e-04 301/             --- Card 6e[2]: C partial phonon DOS
  0.0 3.045983e-06 ... /
  366 367 1/                          --- Card 7: nalpha, nbeta, lat
  ...                                 --- Cards 8-14: alpha, beta, T, rho, etc.
  /
  stop

  In this example:
    - The principal scatterer is C-12 (za=6012)
    - SiC has space group F-43m with 4 Si + 4 C atoms per unit cell
    - CEF selects C as the DC atom (lower σ_inc contribution)
    - The C tape gets LTHR=1 (coherent elastic, scaled by 1/f_DC = 2.0)
    - A separate run with za=14028 gives Si its LTHR=2 (incoherent elastic)
    - For MEF: change Card 6b to "2 2 2/" — both tapes get LTHR=3
"""

import os
# Limit BLAS/OpenMP threads to 1 before numpy import -- the inner loops
# in trans() operate on many small arrays where threading overhead dominates.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from math import sqrt, exp, log, pi, sin, cos, isinf
import sys
import copy

from dataclasses import dataclass
from typing import List, Tuple

# ============================================================================
# Coherent elastic (Bragg-edge) calculator for polycrystalline materials
# (inlined from coherent_elastic_general.py)
#
# Algorithm based on NCrystal:
#   T. Kittelmann et al., Computer Physics Communications 267 (2021) 108082
# ============================================================================

# WL2EKIN = ℏ²/(2 m_n)  [eV·Å²]   (CODATA 2018, from NCrystal NCDefs.hh)
WL2EKIN: float = 0.081804209605330899


@dataclass
class AtomSite:
    """One distinct atom species occupying one or more sites in the unit cell.

    Parameters
    ----------
    b_coh_fm : float
        Coherent scattering length [fm].
    positions : list of (x, y, z)
        Fractional coordinates of each atom of this species in the unit cell.
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
    sites : list of AtomSite
        One entry per distinct atom species.
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
        return sum(len(s.positions) for s in self.sites)

    @property
    def volume(self) -> float:
        """Unit-cell volume [Å³]."""
        ca = np.cos(np.radians(self.alpha))
        cb = np.cos(np.radians(self.beta))
        cg = np.cos(np.radians(self.gamma))
        return (self.a * self.b * self.c *
                np.sqrt(max(0.0, 1.0 - ca**2 - cb**2 - cg**2 + 2.0*ca*cb*cg)))


def _get_reciprocal_lattice_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """Compute the 3×3 reciprocal lattice matrix G.

    G maps integer Miller indices to Cartesian k-vectors:
        k_vec [Å⁻¹] = G @ [h, k, l]
        d-spacing [Å] = 2π / |k_vec|

    Algorithm mirrors NCrystal NCLatticeUtils.cc (getReciprocalLatticeRot).
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
        return np.diag([k2pi / a, k2pi / b, k2pi / c])

    if a90 and b90 and g120:
        sq3 = np.sqrt(3.0)
        return np.array([
            [k2pi / a,             0.0,                    0.0    ],
            [k2pi / (a * sq3),  2.0 * k2pi / (b * sq3),   0.0    ],
            [0.0,               0.0,                    k2pi / c  ],
        ])

    if a90 and g90:
        sb   = np.sin(beta)
        cotb = np.cos(beta) / sb
        return np.array([
            [k2pi / a,            0.0,        0.0          ],
            [0.0,                 k2pi / b,   0.0          ],
            [-cotb * k2pi / a,    0.0,        k2pi/(c*sb)  ],
        ])

    # General triclinic
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


def compute_bragg_edges_general(
    crystal, emax, dcutoff=0.5, fsquarecut=1e-5, merge_tol=1e-4,
):
    """Compute Bragg-edge cross-section data for any polycrystalline material.

    Returns
    -------
    bragg_data : ndarray, shape (N, 2)
        Column 0: Bragg-edge energy E_threshold [eV], ascending.
        Column 1: Per-plane-group cross-section contribution [barn·eV]
                  (Debye-Waller NOT included).
    nbe : int
        Number of rows in bragg_data.
    species_corr : ndarray, shape (nbe, nspecies, nspecies)
        Per-species correlation matrix for Debye-Waller application.
    """
    V = crystal.volume
    N = crystal.n_atoms

    if V <= 0.0:
        raise ValueError(f"Non-positive unit-cell volume {V:.6g} Å³.")
    if N == 0:
        raise ValueError("Crystal has no atom sites.")

    xsectfact = 0.5 * WL2EKIN / (V * N)

    G = _get_reciprocal_lattice_matrix(
        crystal.a, crystal.b, crystal.c,
        crystal.alpha, crystal.beta, crystal.gamma,
    )

    ksq_max = (2.0 * np.pi / dcutoff) ** 2

    h_max = max(1, int(np.ceil(crystal.a / dcutoff)) + 1)
    k_max = max(1, int(np.ceil(crystal.b / dcutoff)) + 1)
    l_max = max(1, int(np.ceil(crystal.c / dcutoff)) + 1)

    sites_data = [
        (s.b_coh_sqrtbarn, np.asarray(s.positions, dtype=float))
        for s in crystal.sites
    ]
    nspecies = len(crystal.sites)

    plane_list = []

    for h in range(0, h_max + 1):
        k_lo = 0 if h == 0 else -k_max
        for k in range(k_lo, k_max + 1):
            l_lo = 1 if (h == 0 and k == 0) else -l_max
            for l in range(l_lo, l_max + 1):

                k_vec = G @ np.array([h, k, l], dtype=float)
                ksq   = float(np.dot(k_vec, k_vec))

                if ksq < 1e-30 or ksq > ksq_max:
                    continue

                d     = 2.0 * np.pi / np.sqrt(ksq)
                E_thr = WL2EKIN / (4.0 * d * d)
                if E_thr > emax:
                    continue

                form_real = np.empty(nspecies)
                form_imag = np.empty(nspecies)
                for si, (_, pos) in enumerate(sites_data):
                    phase = 2.0 * np.pi * (
                        h * pos[:, 0] + k * pos[:, 1] + l * pos[:, 2]
                    )
                    form_real[si] = float(np.sum(np.cos(phase)))
                    form_imag[si] = float(np.sum(np.sin(phase)))

                real_part = 0.0
                imag_part = 0.0
                for si in range(nspecies):
                    real_part += sites_data[si][0] * form_real[si]
                    imag_part += sites_data[si][0] * form_imag[si]

                F2 = real_part**2 + imag_part**2

                if F2 < fsquarecut:
                    continue

                C_st = (np.outer(form_real, form_real) +
                        np.outer(form_imag, form_imag))

                plane_list.append([d, F2, 2.0, C_st])

    if not plane_list:
        return (np.empty((0, 2), dtype=float), 0,
                np.empty((0, nspecies, nspecies), dtype=float))

    plane_list.sort(key=lambda x: -x[0])

    groups = []
    for d, F2, mult, C_st in plane_list:
        merged = False
        for grp in groups:
            d_g, F2_g = grp[0], grp[1]
            if (abs(d   - d_g)  /      d_g         < merge_tol and
                abs(F2  - F2_g) / max(F2_g, 1e-30) < merge_tol):
                grp[2] += mult
                merged = True
                break
        if not merged:
            groups.append([d, F2, mult, C_st])

    pairs = []
    for d, F2, mult, C_st in groups:
        E_thr = WL2EKIN / (4.0 * d * d)
        sigma = d * F2 * mult * xsectfact
        D_st = d * C_st * mult * xsectfact
        pairs.append([E_thr, sigma, D_st])

    pairs.sort(key=lambda p: p[0])

    TOLER = 1e-6
    combined = []
    for E, sig, D_st in pairs:
        if combined and (E - combined[-1][0]) < TOLER:
            combined[-1][1] += sig
            combined[-1][2] = combined[-1][2] + D_st
        else:
            combined.append([E, sig, D_st.copy()])

    bragg_data = np.array([[e[0], e[1]] for e in combined], dtype=float)
    nbe = int(bragg_data.shape[0])

    species_corr = np.zeros((nbe, nspecies, nspecies), dtype=float)
    for j, entry in enumerate(combined):
        species_corr[j] = entry[2]

    if nbe > 0 and bragg_data[-1, 0] < emax:
        bragg_data = np.vstack([bragg_data, [emax, bragg_data[-1, 1]]])
        species_corr = np.concatenate(
            [species_corr, species_corr[-1:]], axis=0)
        nbe += 1

    return bragg_data, nbe, species_corr


# ============================================================================
# Physical constants (matching NJOY2016 physics module exactly)
# ============================================================================
BK = 8.617333262e-5       # Boltzmann constant, eV/K
EV = 1.602176634e-12      # erg/eV
CLIGHT = 2.99792458e10    # cm/s
AMU = 931.49410242e6 * EV / (CLIGHT * CLIGHT)  # g/amu
HBAR = 6.582119569e-16 * EV  # Planck/2pi, erg*s
AMASSN = 1.00866491595     # neutron mass in amu
AMASSP = 1.007276466621    # proton mass in amu
AMASSD = 2.013553212745    # deuteron mass in amu
AMASSE = 5.48579909065e-4  # electron mass in amu

THERM = 0.0253  # thermal energy in eV (for lat=1 scaling)

CARD_END = 'CARD_END'  # Sentinel token marking end of an NJOY input card (/)

# Known NJOY module names (used to detect end of LEAPR input block)
_NJOY_MODULES = {
    'moder', 'reconr', 'broadr', 'unresr', 'heatr', 'thermr', 'groupr',
    'errorr', 'covr', 'acer', 'powr', 'wimsr', 'plotr', 'viewr', 'mixr',
    'dtfr', 'ccccr', 'matxsr', 'resxsr', 'purr', 'gaspr', 'leapr', 'stop',
}

# ============================================================================
# Input parser
# ============================================================================
def parse_leapr_input(filename):
    """Parse a LEAPR input file (NJOY free-format style).

    Returns a dict with all input parameters.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the 'leapr' line
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped == 'leapr':
            start = i + 1
            break

    # Collect all tokens from remaining lines, stopping at module names or 'stop'
    tokens = []

    i = start
    while i < len(lines):
        line = lines[i].strip()

        # Check for module name or stop (end of LEAPR input block)
        if line.lower() in _NJOY_MODULES:
            break

        i += 1
        tokens_in_line = _parse_line(line)
        tokens.extend(tokens_in_line)

    return tokens, lines, start


def _parse_line(line):
    """Parse a single NJOY free-format line into tokens.

    In NJOY free-format input, '/' terminates the current card.
    Values not provided before '/' take their defaults.
    A CARD_END sentinel is inserted at the end of every line to match
    Fortran's record-based I/O (each read() consumes one complete line).
    """
    stripped = line.strip()
    if not stripped:
        return []

    # Check if it's a quoted string line (NJOY supports ' and " as delimiters)
    if stripped.startswith("'") or stripped.startswith('"'):
        quote_char = stripped[0]
        # Find closing quote
        close_idx = stripped.find(quote_char, 1)
        if close_idx > 0:
            text = stripped[:close_idx + 1]
            rest = stripped[close_idx + 1:].strip()
            has_slash = rest.startswith('/')
        else:
            # No closing quote found, take whole line
            text = stripped
            has_slash = False
        result = [('string', text)]
        result.append(CARD_END)
        return result

    # Lines starting with * are comment/title lines in NJOY.
    # Fortran's list-directed `read(nsysi,*) text` reads only the first
    # free-format token (terminated by space, comma, or /).
    # The rest of the line is discarded.
    if stripped.startswith('*'):
        token = ''
        for ch in stripped:
            if ch in (' ', ',', '/'):
                break
            token += ch
        result = [('string', token)]
        result.append(CARD_END)
        return result

    # Check for / card terminator
    has_slash = '/' in stripped
    if has_slash:
        idx = stripped.index('/')
        stripped = stripped[:idx]

    # Split on whitespace and commas
    parts = stripped.replace(',', ' ').split()
    tokens = []
    for p in parts:
        try:
            if '.' not in p and 'e' not in p.lower():
                tokens.append(int(p))
            else:
                tokens.append(float(p))
        except ValueError:
            tokens.append(p)

    # Always insert CARD_END at end of every line to match Fortran's
    # record-based I/O where each read() consumes one complete line.
    tokens.append(CARD_END)

    return tokens


class TokenReader:
    """Sequential reader for parsed tokens with NJOY card boundary support.

    NJOY free-format input uses '/' to terminate cards. When a card provides
    fewer values than expected, the remaining values take their defaults.
    CARD_END sentinels in the token stream mark these boundaries.
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def _consume_card_end(self):
        """Consume remaining tokens on the current card, including CARD_END.

        Matches Fortran's record-based I/O where each read() consumes one
        complete line. Any unread numeric values on the card are discarded.
        """
        while self.pos < len(self.tokens):
            if self.tokens[self.pos] == CARD_END:
                self.pos += 1
                return
            if isinstance(self.tokens[self.pos], (int, float)):
                self.pos += 1  # skip extra values on this card
            else:
                return

    def read_ints(self, n, defaults=None):
        """Read up to n integers from the current card, with defaults.

        Stops at CARD_END and consumes it. Unread values get defaults.
        """
        if defaults is None:
            defaults = [0] * n
        result = list(defaults)
        for i in range(n):
            if self.pos < len(self.tokens) and self.tokens[self.pos] != CARD_END:
                t = self.tokens[self.pos]
                if isinstance(t, (int, float)):
                    result[i] = int(t)
                    self.pos += 1
                else:
                    break
            else:
                break
        self._consume_card_end()
        return result

    def read_floats(self, n, defaults=None):
        """Read up to n floats from the current card, with defaults.

        Stops at CARD_END and consumes it. Unread values get defaults.
        """
        if defaults is None:
            defaults = [0.0] * n
        result = list(defaults)
        for i in range(n):
            if self.pos < len(self.tokens) and self.tokens[self.pos] != CARD_END:
                t = self.tokens[self.pos]
                if isinstance(t, (int, float)):
                    result[i] = float(t)
                    self.pos += 1
                else:
                    break
            else:
                break
        self._consume_card_end()
        return result

    def read_float_array(self, n):
        """Read exactly n floats (possibly spanning multiple lines).

        Skips CARD_END markers between lines since arrays can span multiple
        records. Consumes the trailing CARD_END after the last value.
        """
        result = []
        for _ in range(n):
            # Skip CARD_END markers (line boundaries within multi-line arrays)
            while self.pos < len(self.tokens) and self.tokens[self.pos] == CARD_END:
                self.pos += 1
            if self.pos >= len(self.tokens):
                raise ValueError(f"Unexpected end of input reading array "
                                 f"(got {len(result)} of {n} values)")
            t = self.tokens[self.pos]
            result.append(float(t))
            self.pos += 1
        self._consume_card_end()
        return np.array(result)

    def read_string(self):
        """Read a string token. Consumes trailing CARD_END if present."""
        if self.pos >= len(self.tokens):
            return ''
        t = self.tokens[self.pos]
        self.pos += 1
        if isinstance(t, tuple) and t[0] == 'string':
            self._consume_card_end()
            return t[1]
        self._consume_card_end()
        return str(t)

    def read_comment_strings(self):
        """Read comment string cards until a bare '/' (CARD_END with no data).

        Returns a list of comment strings.

        Handles both quoted string cards (e.g., ' text '/) and unquoted
        cards (e.g., 0/). In Fortran, all are read as character strings
        via read(nsysi,*) text.
        """
        comments = []
        while self.pos < len(self.tokens):
            t = self.tokens[self.pos]
            if isinstance(t, tuple) and t[0] == 'string':
                self.pos += 1
                self._consume_card_end()
                comments.append(t[1])
            elif t == CARD_END:
                # Bare '/' - end of comment section
                self.pos += 1
                break
            else:
                # Non-string token (e.g., numeric '0/')
                # Fortran reads these as character strings too
                val = str(t)
                self.pos += 1
                self._consume_card_end()
                comments.append(val)
        return comments

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None


# ============================================================================
# Core computational routines
# ============================================================================

def fsum(n, p, npt, tau, deltab):
    """Compute integrals over the phonon frequency distribution.

    integral 0 to inf of 2*p*beta**n * hyperbolic dbeta
    where 'hyperbolic' is cosh(tau*beta) for n even, sinh(tau*beta) for n odd.
    """
    arg = deltab * tau / 2.0
    edsq = exp(arg)
    v = 1.0
    an = 1.0 - 2.0 * (n % 2)
    be = 0.0
    fs = 0.0
    w = 1.0

    for ij in range(npt):
        if n > 0:
            w = be ** n
        ff = ((p[ij] * v) * v + (p[ij] * an / v) / v) * w
        if ij == 0 or ij == npt - 1:
            ff = ff / 2.0
        fs += ff
        be += deltab
        v *= edsq

    return fs * deltab


def start(p1, np1, delta1, tev, tbeta):
    """Compute integral functions of the phonon frequency distribution.

    Returns: p (modified spectrum), f0 (DW lambda), tbar, deltab
    """
    npt = np1
    deltab = delta1 / tev
    p = np.copy(p1[:npt])

    # Transform spectrum
    u = deltab
    v = exp(deltab / 2.0)
    p[0] = p[1] / deltab**2
    vv = v
    for j in range(1, npt):
        p[j] = p[j] / (u * (vv - 1.0 / vv))
        vv = v * vv
        u += deltab

    # Calculate normalizing constant
    tau = 0.5
    an = fsum(1, p, npt, tau, deltab)
    an = an / tbeta
    for i in range(npt):
        p[i] = p[i] / an

    # Calculate Debye-Waller lambda and effective temperature
    f0 = fsum(0, p, npt, tau, deltab)
    tbar = fsum(2, p, npt, tau, deltab) / (2.0 * tbeta)

    # Convert p(beta) into t1(beta)
    for i in range(npt):
        be = deltab * i
        p[i] = p[i] * exp(be / 2.0) / f0

    return p, f0, tbar, deltab


def terpt(tn, ntn, delta, be):
    """Interpolate in a table of t_n(beta) for a required beta."""
    if be > ntn * delta:
        return 0.0
    i = int(be / delta)
    if i < ntn - 1:
        bt = i * delta
        btp = bt + delta
        i_idx = i  # 0-based
        return tn[i_idx] + (be - bt) * (tn[i_idx + 1] - tn[i_idx]) / (btp - bt)
    return 0.0


def terpt_vec(tn, ntn, delta, be_arr):
    """Vectorized interpolation in t_n(beta) table for array of beta values."""
    result = np.zeros_like(be_arr)
    i_raw = np.floor(be_arr / delta).astype(int)
    mask = (be_arr >= 0) & (i_raw < ntn - 1)
    if not np.any(mask):
        return result
    be_m = be_arr[mask]
    i = i_raw[mask]
    bt = i * delta
    result[mask] = tn[i] + (be_m - bt) * (tn[i + 1] - tn[i]) / delta
    return result


def convol(t1, tlast, n1, nl, nn, delta):
    """Convolve t1 with tlast to get the next phonon expansion term.

    Returns tnext array and normalization check ckk.
    Uses FFT-based convolution for O(N log N) performance.
    Falls back to scalar loops for small arrays.
    """
    tiny = 1.0e-30

    # Use FFT for large arrays, scalar for small
    # Vectorized convolution: loop over j (n1 is typically small, ~48-301)
    # and compute contributions to all k indices simultaneously using numpy.
    # This avoids the O(n1*nn) Python loop overhead while producing exact
    # results (unlike FFT which has ~1e-16 relative noise floor).
    tnext = np.zeros(nn)
    for j in range(n1):
        if t1[j] <= 0.0:
            continue
        w = 0.5 if (j == 0 or j == n1 - 1) else 1.0
        coeff = w * t1[j]
        exp_be = exp(-j * delta)

        # f1: tnext[k] += coeff * exp(-j*delta) * tlast[k+j]
        #     for k in [0, min(nn, nl-j))
        kmax_f1 = min(nn, nl - j)
        if kmax_f1 > 0:
            tnext[:kmax_f1] += coeff * exp_be * tlast[j:j + kmax_f1]

        # f2 positive (i2 = k-j >= 0): tnext[k] += coeff * tlast[k-j]
        #     for k in [j, min(j+nl, nn))
        klo = j
        khi = min(j + nl, nn)
        if khi > klo:
            tnext[klo:khi] += coeff * tlast[:khi - klo]

        # f2 negative (i2 = k-j < 0): tnext[k] += coeff * exp(-(j-k)*delta) * tlast[j-k]
        #     for k in [max(0, j-nl+1), min(j, nn))
        if j > 0:
            klo2 = max(0, j - nl + 1)
            khi2 = min(j, nn)
            if khi2 > klo2:
                m = j - np.arange(klo2, khi2)  # j-k values (positive)
                tnext[klo2:khi2] += coeff * np.exp(-m * delta) * tlast[m]

    tnext *= delta

    tnext[tnext < tiny] = 0.0

    # Normalization check
    k_arr = np.arange(nn)
    be_k = k_arr * delta
    cc_k = tnext + tnext * np.exp(-be_k)
    cc_k[0] /= 2.0
    if nn > 1:
        cc_k[-1] /= 2.0
    ckk = delta * np.sum(cc_k)

    return tnext, ckk


def contin(ssm_slice, alpha, beta, nalpha, nbeta, lat, arat, tev,
           p1, np1, delta1, tbeta, nphon, iprint, twt, f0_out, tbar_out):
    """Main routine for calculating S(alpha,beta) for continuous distributions.

    ssm_slice: 2D array [nbeta, nalpha] to fill with results
    Returns: f0 (Debye-Waller lambda), tbar (effective temp ratio)
    """
    tiny = 1.0e-30
    explim = -250.0

    sc = 1.0
    if lat == 1:
        sc = THERM / tev

    # Calculate parameters for this temperature
    p, f0, tbar, deltab = start(p1, np1, delta1, tev, tbeta)
    npt = np1

    # Start phonon expansion with t1
    tlast = np.copy(p[:npt])

    # Precompute vectorized quantities
    al_vec = alpha[:nalpha] * sc / arat
    betan = beta[:nbeta] * sc

    # First phonon term: terpt result depends only on beta, not alpha
    st_vec = terpt_vec(p, npt, deltab, betan)  # shape (nbeta,)
    xa = np.log(al_vec * f0)
    ex_vec = -f0 * al_vec + xa
    exx_vec = np.where(ex_vec > explim, np.exp(ex_vec), 0.0)
    # ssm_slice[k, j] = st_vec[k] * exx_vec[j]
    ssm_slice[:nbeta, :nalpha] = st_vec[:, np.newaxis] * exx_vec[np.newaxis, :]
    ssm_slice[ssm_slice < tiny] = 0.0

    npl = npt

    # Track where SCT approximation starts
    maxt = np.full(nbeta, nalpha + 1, dtype=int)

    # Phonon expansion sum
    for n in range(2, nphon + 1):
        npn = npt + npl - 1
        tnow, ckk = convol(p, tlast, npt, npl, npn, deltab)

        # terpt results are independent of alpha
        st_vec = terpt_vec(tnow, npn, deltab, betan)  # shape (nbeta,)
        xa += np.log(al_vec * f0 / n)
        ex_vec = -f0 * al_vec + xa
        exx_vec = np.where(ex_vec > explim, np.exp(ex_vec), 0.0)
        add_matrix = st_vec[:, np.newaxis] * exx_vec[np.newaxis, :]
        add_matrix[add_matrix < tiny] = 0.0
        ssm_slice[:nbeta, :nalpha] += add_matrix

        # Track convergence on last phonon order
        if n >= nphon:
            for k in range(nbeta):
                for j in range(nalpha):
                    if ssm_slice[k, j] != 0.0:
                        if add_matrix[k, j] > ssm_slice[k, j] / 1000.0:
                            if j < maxt[k]:
                                maxt[k] = j

        tlast = np.copy(tnow[:npn])
        npl = npn

    # Apply SCT approximation where phonon expansion hasn't converged
    for k in range(1, nbeta):
        if maxt[k] > maxt[k - 1]:
            maxt[k] = maxt[k - 1]

    for j in range(nalpha):
        al = al_vec[j]
        alw = al * tbeta
        alp = alw * tbar
        for k in range(nbeta):
            if j >= maxt[k]:
                be = betan[k]
                ex = -(alw - be)**2 / (4.0 * alp)
                ssct = 0.0
                if ex > explim:
                    ssct = exp(ex) / sqrt(4.0 * pi * alp)
                ssm_slice[k, j] = ssct

    return f0, tbar, deltab


# ============================================================================
# Translational scattering (diffusion or free gas)
# ============================================================================

def besk1(x):
    """Modified Bessel function K1. Exponential part omitted for x>1."""
    c_coeffs_small = [
        0.442850424, 0.584115288, 6.070134559, 17.864913364,
        48.858995315, 90.924600045, 113.795967431, 85.331474517,
        32.00008698, 3.999998802
    ]
    c_coeffs_small2 = [
        1.304923514, 1.47785657, 16.402802501, 44.732901977,
        115.837493464, 198.437197312, 222.869709703, 142.216613971,
        40.000262262, 1.999996391
    ]

    if x <= 1.0:
        v = 0.125 * x
        u = v * v
        # Horner evaluation: c1*u^9 + c2*u^8 + ... + c10
        bi1 = c_coeffs_small[0]
        for i in range(1, 10):
            bi1 = bi1 * u + c_coeffs_small[i]
        bi1 *= v

        # Horner evaluation: c11*u^9 + c12*u^8 + ... + c20
        bi3 = c_coeffs_small2[0]
        for i in range(1, 10):
            bi3 = bi3 * u + c_coeffs_small2[i]

        return 1.0 / x + bi1 * (log(0.5 * x) + 0.5772156649) - v * bi3
    else:
        u = 1.0 / x
        c_large = [
            0.0108241775, 0.0788000118, 0.2581303765, 0.5050238576,
            0.663229543, 0.6283380681, 0.4594342117, 0.2847618149,
            0.1736431637, 0.1280426636, 0.1468582957, 0.4699927013,
            1.2533141373
        ]
        # Horner: ((((((((((((-c25*u+c26)*u-c27)*u+c28)*u-c29)*u+c30)
        #   *u-c31)*u+c32)*u-c33)*u+c34)*u-c35)*u+c36)*u+c37)
        bi3 = -c_large[0] * u + c_large[1]
        bi3 = bi3 * u - c_large[2]
        bi3 = bi3 * u + c_large[3]
        bi3 = bi3 * u - c_large[4]
        bi3 = bi3 * u + c_large[5]
        bi3 = bi3 * u - c_large[6]
        bi3 = bi3 * u + c_large[7]
        bi3 = bi3 * u - c_large[8]
        bi3 = bi3 * u + c_large[9]
        bi3 = bi3 * u - c_large[10]
        bi3 = bi3 * u + c_large[11]
        bi3 = bi3 * u + c_large[12]
        return sqrt(u) * bi3


def terps(sd, nsd, delta, be):
    """Interpolate in S(alpha,beta) table using log interpolation."""
    slim = -225.0
    if be > delta * nsd:
        return 0.0
    i = int(be / delta)
    if i < nsd - 1:
        bt = i * delta
        btp = bt + delta
        if sd[i] <= 0.0:
            st = slim
        else:
            st = log(sd[i])
        if sd[i + 1] <= 0.0:
            stp = slim
        else:
            stp = log(sd[i + 1])
        stt = st + (be - bt) * (stp - st) / (btp - bt)
        if stt > slim:
            return exp(stt)
    return 0.0


def stable(al, delta, c_diff, twt, iprint, ndmax):
    """Set up table of S-diffusion or S-free.

    Returns sd array and nsd (number of points).
    """
    eps = 1.0e-7
    sd = np.zeros(min(ndmax, 100000))

    if c_diff != 0.0:
        # Diffusion branch
        d = twt * c_diff
        c2 = sqrt(c_diff * c_diff + 0.25)
        c3 = 2.0 * d * al
        c4 = c3 * c3
        c8 = c2 * c3 / pi
        c3_new = 2.0 * d * c_diff * al
        be = 0.0
        j = 0
        idone = False
        while not idone:
            c6 = sqrt(be * be + c4)
            c7 = c6 * c2
            if c7 <= 1.0:
                c5 = c8 * exp(c3_new + be / 2.0)
            else:
                ex = c3_new - c7 + be / 2.0
                c5 = c8 * exp(ex)
            sd[j] = c5 * besk1(c7) / c6
            be += delta
            j += 1
            if j % 2 == 1:  # j is now 1-based count
                if j >= ndmax:
                    idone = True
                if j > 1 and eps * sd[0] >= sd[j - 1]:
                    idone = True
        nsd = j
    else:
        # Free-gas branch
        be = 0.0
        j = 0
        wal = twt * al
        idone = False
        while not idone:
            ex = -(wal - be)**2 / (4.0 * wal)
            sfree = exp(ex) / sqrt(4.0 * pi * wal)
            sd[j] = sfree
            be += delta
            j += 1
            if j % 2 == 1:
                if j >= ndmax:
                    idone = True
                if j > 1 and eps * sd[0] >= sd[j - 1]:
                    idone = True
        nsd = j

    return sd, nsd


def sbfill(nbt, delta, be, s, betan, nbeta, ndmax):
    """Generate s(beta) on a new energy grid for convolution (vectorized)."""
    slim = -225.0
    shade = 1.00001

    bmin = -be - (nbt - 1) * delta
    n_pts = 2 * nbt - 1
    bet_arr = bmin + np.arange(n_pts) * delta
    b_arr = np.abs(bet_arr)

    # Find interpolation indices: j such that betan[j-1] <= b <= betan[j]
    j_arr = np.searchsorted(betan[:nbeta], b_arr, side='right')

    # Boundary handling
    # Top: b > betan[nbeta-1] but within shade → use last interval
    at_top = j_arr >= nbeta
    near_top = at_top & (b_arr < shade * betan[nbeta - 1])
    out_of_range = at_top & ~near_top
    j_arr = np.clip(j_arr, 1, nbeta - 1)

    # Build result with log-interpolation
    sb = np.zeros(2 * nbt + 1)
    valid = ~out_of_range
    if not np.any(valid):
        return sb

    j = j_arr[valid]
    b = b_arr[valid]
    bet = bet_arr[valid]

    log_sj = np.where(s[j] > 0.0, np.log(s[j]), slim)
    log_sjm1 = np.where(s[j - 1] > 0.0, np.log(s[j - 1]), slim)

    sb_val = log_sj + (b - betan[j]) * (log_sjm1 - log_sj) / (betan[j - 1] - betan[j])
    sb_val = np.where(bet > 0.0, sb_val - bet, sb_val)

    result = np.where(sb_val > slim, np.exp(sb_val), 0.0)
    sb[:n_pts][valid] = result
    return sb


def trans(ssm_slice, alpha, beta, nalpha, nbeta, lat, arat, tev,
          twt, c_diff, tbeta, f0, deltab, tbar, iprint):
    """Add translational contribution to S(alpha,beta).

    Modifies ssm_slice in-place.
    """
    tiny = 1.0e-30
    slim = -225.0
    shade = 1.00001
    sc = 1.0
    if lat == 1:
        sc = THERM / tev

    c0, c1_t, c2_t, c3_t, c4_t = 0.4, 1.0, 1.42, 0.2, 10.0
    ndmax = max(nbeta, 1000000)
    betan = beta[:nbeta] * sc
    shade_last = shade * betan[nbeta - 1]

    for ialpha in range(nalpha):
        al = alpha[ialpha] * sc / arat

        # Choose beta interval for convolution
        ded = c0 * (twt * c_diff * al) / sqrt(c1_t + c2_t * (twt * c_diff * al) * c_diff)
        if ded == 0.0:
            ded = c3_t * sqrt(twt * al)
        deb = c4_t * al * deltab
        delta = min(deb, ded)

        # Make table of s-diffusion or s-free
        sd, nsd = stable(al, delta, c_diff, twt, iprint, ndmax)

        if nsd > 1:
            ap = ssm_slice[:nbeta, ialpha].copy()
            nbt = nsd
            n_pts = 2 * nbt - 1

            # Precompute log(ap) and interpolation slopes ONCE per alpha
            log_ap = np.full(nbeta, slim)
            pos_mask = ap > 0.0
            log_ap[pos_mask] = np.log(ap[pos_mask])
            # slope[k] = (log_ap[k] - log_ap[k+1]) / (betan[k] - betan[k+1])
            slope = (log_ap[:-1] - log_ap[1:]) / (betan[:-1] - betan[1:])

            # Simpson's weights * sd
            i_arr = np.arange(nbt)
            f_arr = 2.0 * ((i_arr % 2) + 1).astype(float)
            f_arr[0] = 1.0
            if nbt > 1:
                f_arr[-1] = 1.0
            fsd = f_arr * sd[:nbt]
            fsd_bwd = fsd * np.exp(-i_arr * delta)

            exp_alf0 = exp(-al * f0)
            j_grid = np.arange(n_pts, dtype=np.float64) * delta
            nbt_m1_delta = (nbt - 1) * delta
            delta_nsd = delta * nsd

            # Pre-allocate buffers for inner loop
            bet_arr = np.empty(n_pts)
            b_arr = np.empty(n_pts)
            sb_val = np.empty(n_pts)
            sb = np.empty(n_pts)

            for ibeta in range(nbeta):
                be = betan[ibeta]
                bmin = -be - nbt_m1_delta

                np.add(j_grid, bmin, out=bet_arr)
                np.abs(bet_arr, out=b_arr)

                j_arr = np.searchsorted(betan, b_arr, side='right')
                out_mask = (j_arr >= nbeta) & (b_arr >= shade_last)
                np.clip(j_arr, 1, nbeta - 1, out=j_arr)

                # Compute interpolation for ALL points (avoids boolean indexing)
                jm1 = j_arr - 1
                np.subtract(b_arr, betan[j_arr], out=sb_val)
                np.multiply(sb_val, slope[jm1], out=sb_val)
                np.add(sb_val, log_ap[j_arr], out=sb_val)
                # Subtract bet for positive beta points
                pos = bet_arr > 0.0
                sb_val[pos] -= bet_arr[pos]
                # Convert log->linear, zero out-of-range and below-threshold
                good = (~out_mask) & (sb_val > slim)
                sb[:] = 0.0
                np.exp(sb_val, where=good, out=sb)

                # Convolution via dot products
                s = (np.dot(fsd, sb[nbt - 1:2 * nbt - 1])
                     + np.dot(fsd_bwd, sb[nbt - 1::-1]))
                s *= delta / 3.0
                if s < tiny:
                    s = 0.0

                # terps: interpolate in sd table
                if be <= delta_nsd:
                    i_idx = int(be / delta)
                    if i_idx < nsd - 1:
                        bt = i_idx * delta
                        sd_i = sd[i_idx]
                        sd_ip1 = sd[i_idx + 1]
                        log_sd_i = log(sd_i) if sd_i > 0.0 else slim
                        log_sd_ip1 = log(sd_ip1) if sd_ip1 > 0.0 else slim
                        stt = log_sd_i + (be - bt) * (log_sd_ip1 - log_sd_i) / delta
                        st = exp(stt) if stt > slim else 0.0
                        if st > 0.0:
                            s += exp_alf0 * st
                            if s < tiny:
                                s = 0.0

                ssm_slice[ibeta, ialpha] = s


# ============================================================================
# Discrete oscillators
# ============================================================================

def bfact(x, dwc, betai):
    """Calculate Bessel function terms for discrete oscillators.

    Returns bzero, bplus[50], bminus[50].
    """
    big = 1.0e10
    tiny = 1.0e-30
    imax = 50

    # Compute bessi0
    y = x / 3.75
    if y <= 1.0:
        u = y * y
        bessi0 = (1.0 + u * (3.5156229 + u * (3.0899424 + u * (1.2067492 +
                  u * (0.2659732 + u * (0.0360768 + u * 0.0045813))))))
    else:
        v = 1.0 / y
        bessi0 = (0.39894228 + v * (0.01328592 + v * (0.00225319 +
                  v * (-0.00157565 + v * (0.00916281 + v * (-0.02057706 +
                  v * (0.02635537 + v * (-0.01647633 + v * 0.00392377)))))))) / sqrt(x)

    # Compute bessi1
    if y <= 1.0:
        u = y * y
        bessi1 = (0.5 + u * (0.87890594 + u * (0.51498869 + u * (0.15084934 +
                  u * (0.02658733 + u * (0.00301532 + u * 0.00032411)))))) * x
    else:
        v = 1.0 / y
        bessi1 = 0.02282967 + v * (-0.02895312 + v * (0.01787654 - v * 0.00420059))
        bessi1 = (0.39894228 + v * (-0.03988024 + v * (-0.00362018 +
                  v * (0.00163801 + v * (-0.01031555 + v * bessi1))))) / sqrt(x)

    # Generate higher orders by reverse recursion
    bn = np.zeros(imax)
    bn[imax - 1] = 0.0
    bn[imax - 2] = 1.0
    i = imax - 2
    while i > 0:
        bn[i - 1] = bn[i + 1] + (i + 1) * (2.0 / x) * bn[i]
        i -= 1
        if bn[i] >= big:
            bn[i:] /= big

    rat = bessi1 / bn[0]
    bn *= rat
    # Clean up inf/NaN from recursion overflow (matches Fortran behavior where
    # such values propagate but are filtered by comparison checks)
    bn[~np.isfinite(bn)] = 0.0
    bn[bn < tiny] = 0.0

    # Apply exponential terms
    bplus = np.zeros(imax)
    bminus = np.zeros(imax)

    if y <= 1.0:
        bzero = bessi0 * exp(-dwc)
        for i in range(imax):
            if bn[i] != 0.0:
                arg = -dwc - (i + 1) * betai / 2.0
                if arg > 709.0:
                    bplus[i] = 0.0
                else:
                    bplus[i] = exp(arg) * bn[i]
                    if bplus[i] < tiny:
                        bplus[i] = 0.0
                arg = -dwc + (i + 1) * betai / 2.0
                if arg > 709.0:
                    bminus[i] = 0.0
                else:
                    bminus[i] = exp(arg) * bn[i]
                    if bminus[i] < tiny:
                        bminus[i] = 0.0
    else:
        bzero = bessi0 * exp(-dwc + x)
        for i in range(imax):
            if bn[i] != 0.0:
                arg = -dwc - (i + 1) * betai / 2.0 + x
                if arg > 709.0:
                    bplus[i] = 0.0
                else:
                    bplus[i] = exp(arg) * bn[i]
                    if bplus[i] < tiny:
                        bplus[i] = 0.0
                arg = -dwc + (i + 1) * betai / 2.0 + x
                if arg > 709.0:
                    bminus[i] = 0.0
                else:
                    bminus[i] = exp(arg) * bn[i]
                    if bminus[i] < tiny:
                        bminus[i] = 0.0

    return bzero, bplus, bminus


def bfill(betan, nbeta):
    """Set up bex and rdbex arrays for sint.

    Creates the extended beta grid [-beta_max, ..., 0, ..., +beta_max].
    """
    small = 1.0e-9

    bex = []
    # Negative betas (reversed)
    for i in range(nbeta - 1, -1, -1):
        bex.append(-betan[i])

    # Handle beta=0 case
    if betan[0] <= small:
        bex[-1] = 0.0
        k = nbeta
    else:
        bex.append(betan[0])
        k = nbeta + 1

    # Positive betas
    for i in range(1, nbeta):
        bex.append(betan[i])
        k += 1

    bex = np.array(bex)
    nbx = len(bex)
    rdbex = np.zeros(nbx - 1)
    for i in range(nbx - 1):
        rdbex[i] = 1.0 / (bex[i + 1] - bex[i])

    return bex, rdbex, nbx


def exts(sexpb, exb, betan, nbeta):
    """Set up the sex array for sint.

    Extends asymmetric sab to plus and minus beta.
    """
    small = 1.0e-9
    sex = []

    # Negative beta side (reversed)
    for i in range(nbeta - 1, -1, -1):
        sex.append(sexpb[i])

    if betan[0] <= small:
        sex[-1] = sexpb[0]
        k = nbeta
    else:
        sex.append(sexpb[0])
        k = nbeta + 1

    # Positive beta side
    for i in range(1, nbeta):
        sex.append(sexpb[i] * exb[i] * exb[i])
        k += 1

    return np.array(sex)


def sint(x, bex, rdbex, sex, nbx, alph, wt, tbart, betan, nbeta):
    """Interpolate in scattering function or use SCT approximation."""
    slim = -225.0

    # SCT approximation for |x| > beta_max
    if abs(x) > betan[nbeta - 1]:
        if alph <= 0.0:
            return 0.0
        ex = -(wt * alph - abs(x))**2 / (4.0 * wt * alph * tbart)
        if x > 0.0:
            ex -= x
        return exp(ex) / (4.0 * pi * wt * alph * tbart)

    # Binary search for x in bex
    k1 = 0
    k2 = nbeta - 1
    k3 = nbx - 1

    idone = False
    while not idone:
        if x == bex[k2]:
            return sex[k2]
        elif x > bex[k2]:
            k1 = k2
            k2 = (k3 - k2) // 2 + k2
            if k3 - k1 <= 1:
                idone = True
        else:
            k3 = k2
            k2 = (k2 - k1) // 2 + k1
            if k3 - k1 <= 1:
                idone = True

    if sex[k1] <= 0.0:
        ss1 = slim
    else:
        ss1 = log(sex[k1])
    if sex[k3] <= 0.0:
        ss3 = slim
    else:
        ss3 = log(sex[k3])

    ex = ((bex[k3] - x) * ss1 + (x - bex[k1]) * ss3) * rdbex[k1]
    if ex > slim:
        return exp(ex)
    return 0.0


def sint_vec(x_arr, bex, rdbex, sex, nbx, alph, wt, tbart, betan, nbeta):
    """Vectorized interpolation in scattering function or SCT approximation."""
    slim = -225.0
    result = np.zeros(len(x_arr))
    beta_max = betan[nbeta - 1]

    # SCT approximation for |x| > beta_max
    sct_mask = np.abs(x_arr) > beta_max
    if np.any(sct_mask) and alph > 0.0:
        xs = x_arr[sct_mask]
        ex = -(wt * alph - np.abs(xs))**2 / (4.0 * wt * alph * tbart)
        ex = np.where(xs > 0.0, ex - xs, ex)
        result[sct_mask] = np.exp(ex) / (4.0 * pi * wt * alph * tbart)

    # Interpolation for |x| <= beta_max
    interp_mask = ~sct_mask
    if np.any(interp_mask):
        xs = x_arr[interp_mask]
        # Use searchsorted to find k3 (upper bracket index in bex)
        k3 = np.searchsorted(bex[:nbx], xs, side='right')
        k3 = np.clip(k3, 1, nbx - 1)
        k1 = k3 - 1

        ss1 = np.where(sex[k1] > 0.0, np.log(sex[k1]), slim)
        ss3 = np.where(sex[k3] > 0.0, np.log(sex[k3]), slim)

        ex = ((bex[k3] - xs) * ss1 + (xs - bex[k1]) * ss3) * rdbex[k1]
        result[interp_mask] = np.where(ex > slim, np.exp(ex), 0.0)

    return result


def discre(ssm_slice, alpha, beta, nalpha, nbeta, lat, arat, tev,
           twt, tbeta, nd, bdel, adel, dwpix_val, tempf_val, tempr_val,
           iprint):
    """Convolve discrete oscillators with continuous S(alpha,beta).

    Modifies ssm_slice in-place.
    Returns: updated dwpix, updated tempf
    """
    small = 1.0e-8
    vsmall = 1.0e-10
    tiny = 1.0e-20
    maxdd = 500

    sc = 1.0
    if lat == 1:
        sc = THERM / tev

    # Set up oscillator parameters
    bdeln = np.zeros(nd)
    eb = np.zeros(nd)
    ar = np.zeros(nd)
    dbw = np.zeros(nd)
    dist = np.zeros(nd)
    dwt = 0.0

    for i in range(nd):
        bdeln[i] = bdel[i] / tev
        dwt += adel[i]

    tsave = 0.0
    dw0 = dwpix_val

    for i in range(nd):
        eb[i] = exp(bdeln[i] / 2.0)
        sn = (eb[i] - 1.0 / eb[i]) / 2.0
        cn = (eb[i] + 1.0 / eb[i]) / 2.0
        ar[i] = adel[i] / (sn * bdeln[i])
        dist[i] = adel[i] * bdel[i] * cn / (2.0 * sn)
        tsave += dist[i] / BK
        dbw[i] = ar[i] * cn
        if dwpix_val > 0.0:
            dwpix_val += dbw[i]

    # Prepare functions of beta
    betan = beta[:nbeta] * sc
    exb = np.exp(-betan / 2.0)

    bex, rdbex, nbx = bfill(betan, nbeta)
    wt = tbeta
    tbart = tempf_val / tempr_val

    # Main alpha loop
    for nal in range(nalpha):
        al = alpha[nal] * sc / arat
        dwf = exp(-al * dw0)
        sex = exts(ssm_slice[:, nal], exb, betan, nbeta)
        sexpb = np.zeros(nbeta)

        # Initialize delta function calculation
        ben = np.zeros(maxdd)
        wtn = np.zeros(maxdd)
        ben[0] = 0.0
        wtn[0] = 1.0
        nn = 1
        n = 0

        bes = np.zeros(maxdd)
        wts = np.zeros(maxdd)

        wt_local = tbeta
        tbart_local = tempf_val / tempr_val

        # Loop over all oscillators
        for i_osc in range(nd):
            dwc = al * dbw[i_osc]
            x = al * ar[i_osc]
            bzero, bplus, bminus = bfact(x, dwc, bdeln[i_osc])

            n = 0

            # n=0 term
            for m in range(nn):
                besn = ben[m]
                wtsn = wtn[m] * bzero
                if besn <= 0.0 or wtsn >= small:
                    if n < maxdd:
                        bes[n] = besn
                        wts[n] = wtsn
                        n += 1

            # Negative n terms
            k = 0
            idone = False
            while k < 50 and not idone:
                if bminus[k] <= 0.0:
                    idone = True
                else:
                    for m in range(nn):
                        besn = ben[m] - (k + 1) * bdeln[i_osc]
                        wtsn = wtn[m] * bminus[k]
                        if wtsn >= small and n < maxdd:
                            bes[n] = besn
                            wts[n] = wtsn
                            n += 1
                k += 1

            # Positive n terms
            k = 0
            idone = False
            while k < 50 and not idone:
                if bplus[k] <= 0.0:
                    idone = True
                else:
                    for m in range(nn):
                        besn = ben[m] + (k + 1) * bdeln[i_osc]
                        wtsn = wtn[m] * bplus[k]
                        if wtsn >= small and n < maxdd:
                            bes[n] = besn
                            wts[n] = wtsn
                            n += 1
                k += 1

            # Update for next oscillator
            nn = n
            ben[:nn] = bes[:nn].copy()
            wtn[:nn] = wts[:nn].copy()
            n = 0
            wt_local += adel[i_osc]
            tbart_local += dist[i_osc] / BK / tempr_val

        n = nn

        # Sort discrete lines by weight (descending), skip element 0 (zero state)
        # Fortran sorts elements 2..n (1-based), equivalent to 1..n-1 (0-based)
        if n > 2:
            wts_sub = wts[1:n].copy()
            bes_sub = bes[1:n].copy()
            sort_idx = np.argsort(-wts_sub)
            wts[1:n] = wts_sub[sort_idx]
            bes[1:n] = bes_sub[sort_idx]

        # Trim small entries (Fortran: nn=n-1, loop i=1..nn, n=i)
        # Always drops at least the last (smallest) element
        # When nn=0, the Fortran loop doesn't execute and n stays unchanged
        nn = n - 1
        if nn > 0:
            n_new = nn
            for i in range(nn):  # 0-based, maps to Fortran i=1..nn
                n_new = i + 1
                if wts[i] < 100 * small and (i + 1) > 5:
                    break
            n = n_new

        # Add the continuum part to the scattering law (vectorized over beta)
        for m in range(n):
            be_arr = -betan - bes[m]
            st_arr = sint_vec(be_arr, bex, rdbex, sex, nbx, al, tbeta + twt,
                             tbart_local, betan, nbeta)
            add_arr = wts[m] * st_arr
            add_arr[add_arr < tiny] = 0.0
            sexpb += add_arr

        # Add the delta functions to the scattering law
        if twt <= 0.0:
            m = 0
            idone = False
            while m < n and not idone:
                if dwf < vsmall:
                    idone = True
                else:
                    if bes[m] < 0.0:
                        be = -bes[m]
                        if be <= betan[nbeta - 2]:
                            db = 1000.0
                            done2 = False
                            j = 0
                            jj = 0
                            while j < nbeta and not done2:
                                jj = j
                                if abs(be - betan[j]) > db:
                                    done2 = True
                                else:
                                    db = abs(be - betan[j])
                                j += 1

                            if jj <= 1:
                                add = wts[m] / betan[jj]
                            else:
                                add = 2.0 * wts[m] / (betan[jj] - betan[jj - 2])
                            add *= dwf
                            if add >= tiny:
                                sexpb[jj - 1] += add
                m += 1

        # Record results
        ssm_slice[:, nal] = sexpb

    # Update effective temperature and Debye-Waller
    tempf_new = (tbeta + twt) * tempf_val + tsave

    return dwpix_val, tempf_new


# ============================================================================
# Cold hydrogen/deuterium
# ============================================================================

def sjbes(n, x):
    """Spherical Bessel functions for cold hydrogen calculation."""
    if x < 0.0 or n < 0:
        return 0.0

    if n >= 30000 or x > 3.0e4:
        return 0.0

    huge = 1.0e25
    small_val = 2.0e-38

    if x <= 7.0e-4:
        w = 1.0
        if n == 0:
            return w
        elif n > 10:
            return 0.0
        else:
            t1 = 3.0
            t2 = 1.0
            for i in range(n):
                t3 = t2 * x / t1
                t1 += 2.0
                t2 = t3
            return t3

    if x < 0.2:
        y = x * x
        w = 1.0 - y * (1.0 - y / 20.0) / 6.0
    else:
        w = sin(x) / x

    if n == 0:
        return w

    if x >= 100.0:
        l = int(x / 50.0 + 18)
    elif x >= 10.0:
        l = int(x / 10.0 + 10)
    elif x > 1.0:
        l = int(x / 2.0 + 5)
    else:
        l = 5

    iii = int(x)
    kmax = max(n, iii)
    nm = kmax + l
    z = 1.0 / x
    t3 = 0.0
    t2 = small_val
    sj = 0.0

    for i in range(nm, 0, -1):
        k = i - 1
        t1 = (2 * k + 3) * z * t2 - t3
        if n == k:
            sj = t1
        if abs(t1) >= huge:
            t1 /= huge
            t2 /= huge
            sj /= huge
        t3 = t2
        t2 = t1

    return w * sj / t1


def cn_cg(jj, ll, nn):
    """Clebsch-Gordon coefficients for cold hydrogen."""
    kdet = (jj + ll + nn) // 2
    kdel = jj + ll + nn - 2 * kdet

    if kdel != 0:
        return 0.0

    ka1 = jj + ll + nn
    ka2 = jj + ll - nn
    ka3 = jj - ll + nn
    ka4 = ll - jj + nn

    if ka2 < 0 or ka3 < 0 or ka4 < 0:
        return 0.0

    kb1 = ka1 // 2
    kb2 = ka2 // 2
    kb3 = ka3 // 2
    kb4 = ka4 // 2

    def log_factorial(n):
        s = 0.0
        for i in range(1, n + 1):
            s += log(float(i))
        return s

    lf_ka1 = log_factorial(ka1)
    lf_ka2 = log_factorial(ka2)
    lf_ka3 = log_factorial(ka3)
    lf_ka4 = log_factorial(ka4)
    lf_kb1 = log_factorial(kb1)
    lf_kb2 = log_factorial(kb2)
    lf_kb3 = log_factorial(kb3)
    lf_kb4 = log_factorial(kb4)

    a1 = exp(lf_ka1 / 2.0) if lf_ka1 > 0.0 else 1.0
    a2 = exp(lf_ka2 / 2.0) if lf_ka2 > 0.0 else 1.0
    a3 = exp(lf_ka3 / 2.0) if lf_ka3 > 0.0 else 1.0
    a4 = exp(lf_ka4 / 2.0) if lf_ka4 > 0.0 else 1.0
    b1 = exp(lf_kb1) if lf_kb1 > 0.0 else 1.0
    b2 = exp(lf_kb2) if lf_kb2 > 0.0 else 1.0
    b3 = exp(lf_kb3) if lf_kb3 > 0.0 else 1.0
    b4 = exp(lf_kb4) if lf_kb4 > 0.0 else 1.0

    rat = (2 * nn + 1) / (jj + ll + nn + 1)
    iwign = (jj + ll - nn) // 2
    wign = (-1)**iwign * sqrt(rat) * b1 / a1 * a2 / b2 * a3 / b3 * a4 / b4

    return wign


def sumh(j, jp, y):
    """Sum over Bessel functions and Clebsch-Gordon coefficients."""
    if j == 0:
        return (sjbes(jp, y) * cn_cg(j, jp, jp))**2
    elif jp == 0:
        return (sjbes(j, y) * cn_cg(j, 0, j))**2
    else:
        sum1 = 0.0
        imk = abs(j - jp)
        ipk1 = j + jp + 1
        mpk = ipk1 - imk
        if mpk <= 9:
            ipk = ipk1
        else:
            ipk = imk + 9
        for n in range(imk, ipk):
            sum1 += (sjbes(n, y) * cn_cg(j, jp, n))**2
        return sum1


def bt_stat(j, x):
    """Statistical weight factor for cold hydrogen/deuterium."""
    yy = 0.5 * j * (j + 1)
    a = (2 * j + 1) * exp(-yy * x)
    b = 0.0
    for i in range(10):
        k = 2 * i
        if j % 2 == 1:
            k += 1
        yy = 0.5 * k * (k + 1)
        b += (2 * k + 1) * exp(-yy * x)
    return a / (2.0 * b)


def terpk(ska, nka, delta, be):
    """Interpolate in ska(kappa) table."""
    if be > nka * delta:
        return 1.0
    i = int(be / delta)
    if i < nka - 1:
        bt_val = i * delta
        btp = bt_val + delta
        return ska[i] + (be - bt_val) * (ska[i + 1] - ska[i]) / (btp - bt_val)
    return 1.0


def coldh(ssm_slice, ssp_slice, alpha, beta, nalpha, nbeta, lat, arat, tev,
          twt, tbeta, ncold, ska, nka, dka, tempf_val, tempr_val, iprint):
    """Convolve S(alpha,beta) with rotational modes for cold H2/D2.

    Modifies ssm_slice and ssp_slice in-place.
    """
    pmass = 1.6726231e-24
    dmass = 3.343586e-24
    deh = 0.0147
    ded = 0.0074
    sampch = 0.356
    sampcd = 0.668
    sampih = 2.526
    sampid = 0.403
    small_val = 1.0e-6
    angst = 1.0e-8
    jterm = 3

    sc = 1.0
    if lat == 1:
        sc = THERM / tev

    law = ncold + 1
    de = deh
    if law > 3:
        de = ded
    x = de / tev

    if law > 3:
        amassm = 6.69e-24
        sampc = sampcd
        bp = HBAR / 2.0 * sqrt(2.0 / ded / EV / dmass) / angst
        sampi = sampid
    else:
        amassm = 3.3464e-24
        sampc = sampch
        bp = HBAR / 2.0 * sqrt(2.0 / deh / EV / pmass) / angst
        sampi = sampih

    wt = twt + tbeta
    tbart = tempf_val / tempr_val

    # Prepare arrays
    betan = beta[:nbeta].copy()
    if lat == 1:
        betan *= THERM / tev
    exb = np.exp(-betan / 2.0)
    bex, rdbex, nbx = bfill(betan, nbeta)

    for nal in range(nalpha):
        al = alpha[nal] * sc / arat
        alp = wt * al
        waven = angst * sqrt(amassm * tev * EV * al) / HBAR
        y = bp * waven

        sk = terpk(ska, nka, dka, waven)

        # Spin-correlation factors
        snorm = sampi**2 + sampc**2
        if law == 2:
            swe = sampi**2 / 3.0
            swo = sk * sampc**2 + 2.0 * sampi**2 / 3.0
        elif law == 3:
            swe = sk * sampc**2
            swo = sampi**2
        elif law == 4:
            swe = sk * sampc**2 + 5.0 * sampi**2 / 8.0
            swo = 3.0 * sampi**2 / 8.0
        elif law == 5:
            swe = 3.0 * sampi**2 / 4.0
            swo = sk * sampc**2 + sampi**2 / 4.0
        swe /= snorm
        swo /= snorm

        sex = exts(ssm_slice[:, nal], exb, betan, nbeta)

        # Loop over all beta values (positive and negative)
        jjmax = 2 * nbeta - 1
        for jj in range(jjmax):
            if jj < nbeta - 1:
                k = nbeta - jj - 1  # 0-based: Fortran k=nbeta-jj+1 → 0-based nbeta-jj-1
            else:
                k = jj - nbeta + 1  # 0-based
            be = betan[k]
            if jj < nbeta - 1:
                be = -be
            sn = 0.0

            # Loop over oscillators
            ipo = 1
            if law == 2 or law == 5:
                ipo = 2
            jt1 = 2 * jterm
            if ipo == 2:
                jt1 += 1

            for l in range(ipo, jt1 + 1, 2):
                j = l - 1
                pj = bt_stat(j, x)

                # Sum over even j-prime
                snlg = 0.0
                for lp in range(1, 11, 2):
                    jp = lp - 1
                    betap = (-j * (j + 1) + jp * (jp + 1)) * x / 2.0
                    tmp = (2 * jp + 1) * pj * swe * 4.0 * sumh(j, jp, y)
                    bn = be + betap
                    add = sint(bn, bex, rdbex, sex, nbx, al, wt, tbart,
                              betan, nbeta)
                    snlg += tmp * add

                # Sum over odd j-prime
                snlk = 0.0
                for lp in range(2, 11, 2):
                    jp = lp - 1
                    betap = (-j * (j + 1) + jp * (jp + 1)) * x / 2.0
                    tmp = (2 * jp + 1) * pj * swo * 4.0 * sumh(j, jp, y)
                    bn = be + betap
                    add = sint(bn, bex, rdbex, sex, nbx, al, wt, tbart,
                              betan, nbeta)
                    snlk += tmp * add

                sn += snlg + snlk

            # Store results
            if jj < nbeta:
                ssm_slice[k, nal] = sn
            if jj >= nbeta - 1:
                ssp_slice[k, nal] = sn


# ============================================================================
# Coherent elastic (Bragg edges)
# ============================================================================

def coher(lat, natom, emax):
    """Compute Bragg energies and structure factors for coherent elastic.

    Returns: bragg array of (energy, structure_factor) pairs, nedge
    """
    twopis = (2.0 * pi)**2
    amne = AMASSN * AMU
    econ = EV * 8.0 * (amne / HBAR) / HBAR
    recon = 1.0 / econ
    tsqx = econ / 20.0
    eps = 0.05
    toler = 1.0e-6

    # Material constants
    if lat == 1:  # Graphite
        a, c = 2.4573e-8, 6.700e-8
        amsc, scoh = 12.011, 5.50 / natom
    elif lat == 2:  # Beryllium
        a, c = 2.2856e-8, 3.5832e-8
        amsc, scoh = 9.01, 7.53 / natom
    elif lat == 3:  # BeO
        a, c = 2.695e-8, 4.39e-8
        amsc, scoh = 12.5, 1.0 / natom
    elif lat == 4:  # Aluminum
        a = 4.04e-8
        amsc, scoh = 26.7495, 1.495 / natom
    elif lat == 5:  # Lead
        a = 4.94e-8
        amsc, scoh = 207.0, 1.0 / natom
    elif lat == 6:  # Iron
        a = 2.86e-8
        amsc, scoh = 55.454, 12.9 / natom

    if lat < 4:
        c1 = 4.0 / (3.0 * a * a)
        c2 = 1.0 / (c * c)
        sqrt3 = 1.732050808
        scon = scoh * (4.0 * pi)**2 / (2.0 * a * a * c * sqrt3 * econ)
    elif lat <= 5:
        c1 = 3.0 / (a * a)
        scon = scoh * (4.0 * pi)**2 / (16.0 * a * a * a * econ)
    else:
        c1 = 2.0 / (a * a)
        scon = scoh * (4.0 * pi)**2 / (8.0 * a * a * a * econ)

    wint = 0.0
    t2 = HBAR / (2.0 * AMU * amsc)
    ulim = econ * emax

    # Store edges as parallel arrays (matching Fortran's b array)
    b_tsq = []
    b_f = []
    k = 0  # number of edges found so far

    if lat < 4:
        # Hexagonal lattice: within-loop merge matching Fortran exactly.
        # For small tsq (<= tsqx), edges are added without merging.
        # For larger tsq, scan ALL existing edges to find a merge candidate.
        phi = ulim / twopis
        i1m = int(a * sqrt(phi)) + 1

        for i1 in range(1, i1m + 1):
            l1 = i1 - 1
            i2m = int((l1 + sqrt(3.0 * (a * a * phi - l1 * l1))) / 2.0) + 1

            for i2 in range(i1, i2m + 1):
                l2 = i2 - 1
                x = phi - c1 * (l1 * l1 + l2 * l2 - l1 * l2)
                i3m = 0
                if x > 0:
                    i3m = int(c * sqrt(x))
                i3m += 1

                for i3 in range(1, i3m + 1):
                    l3 = i3 - 1
                    w1 = 2.0 if l1 != l2 else 1.0
                    w2 = 2.0
                    if l1 == 0 or l2 == 0:
                        w2 = 1.0
                    if l1 == 0 and l2 == 0:
                        w2 = w2 / 2.0
                    w3 = 2.0 if l3 != 0 else 1.0

                    # Positive l2
                    tsq = (c1 * (l1 * l1 + l2 * l2 + l1 * l2) + l3 * l3 * c2) * twopis
                    if tsq > 0.0 and tsq <= ulim:
                        tau = sqrt(tsq)
                        w = exp(-tsq * t2 * wint) * w1 * w2 * w3 / tau
                        f = w * formf(lat, l1, l2, l3)
                        if k <= 0 or tsq <= tsqx:
                            b_tsq.append(tsq)
                            b_f.append(f)
                            k += 1
                        else:
                            idone = False
                            for ii in range(k):
                                if tsq >= b_tsq[ii] and tsq < (1 + eps) * b_tsq[ii]:
                                    b_f[ii] += f
                                    idone = True
                                    break
                            if not idone:
                                b_tsq.append(tsq)
                                b_f.append(f)
                                k += 1

                    # Negative l2
                    tsq = (c1 * (l1 * l1 + l2 * l2 - l1 * l2) + l3 * l3 * c2) * twopis
                    if tsq > 0.0 and tsq <= ulim:
                        tau = sqrt(tsq)
                        w = exp(-tsq * t2 * wint) * w1 * w2 * w3 / tau
                        f = w * formf(lat, l1, -l2, l3)
                        if k <= 0 or tsq <= tsqx:
                            b_tsq.append(tsq)
                            b_f.append(f)
                            k += 1
                        else:
                            idone = False
                            for ii in range(k):
                                if tsq >= b_tsq[ii] and tsq < (1 + eps) * b_tsq[ii]:
                                    b_f[ii] += f
                                    idone = True
                                    break
                            if not idone:
                                b_tsq.append(tsq)
                                b_f.append(f)
                                k += 1

    elif lat <= 5:
        # FCC lattice
        i1m = 15
        twothd = 2.0 / 3.0
        for i1 in range(-i1m, i1m + 1):
            for i2 in range(-i1m, i1m + 1):
                for i3 in range(-i1m, i1m + 1):
                    tsq = c1 * (i1*i1 + i2*i2 + i3*i3 + twothd*i1*i2 +
                                twothd*i1*i3 - twothd*i2*i3) * twopis
                    if tsq > 0.0 and tsq <= ulim:
                        tau = sqrt(tsq)
                        w = exp(-tsq * t2 * wint) / tau
                        f = w * formf(lat, i1, i2, i3)
                        b_tsq.append(tsq)
                        b_f.append(f)
                        k += 1

    else:
        # BCC lattice
        i1m = 15
        for i1 in range(-i1m, i1m + 1):
            for i2 in range(-i1m, i1m + 1):
                for i3 in range(-i1m, i1m + 1):
                    tsq = c1 * (i1*i1 + i2*i2 + i3*i3 + i1*i2 + i2*i3 + i1*i3) * twopis
                    if tsq > 0.0 and tsq <= ulim:
                        tau = sqrt(tsq)
                        w = exp(-tsq * t2 * wint) / tau
                        f = w * formf(lat, i1, i2, i3)
                        b_tsq.append(tsq)
                        b_f.append(f)
                        k += 1

    if k == 0:
        return np.array([]), 0

    # Sort by tsq (selection sort matching Fortran)
    pairs = sorted(zip(b_tsq[:k], b_f[:k]))
    b_tsq = [p[0] for p in pairs]
    b_f = [p[1] for p in pairs]

    kept_tsq = b_tsq
    kept_f = b_f

    # Add final edge at ulim
    kept_tsq.append(ulim)
    kept_f.append(kept_f[-1])
    k += 1

    # Convert to practical units and combine duplicate Bragg edges
    bragg = []
    bel = -1.0
    for i in range(k):
        be = kept_tsq[i] * recon
        bs = kept_f[i] * scon
        if be - bel < toler and bragg:
            bragg[-1] = (bragg[-1][0], bragg[-1][1] + bs)
        else:
            bragg.append((be, bs))
            bel = be

    return bragg, len(bragg)


def formf(lat, l1, l2, l3):
    """Compute form factors for the specified lattice."""
    if lat == 1:  # Graphite
        if l3 % 2 != 0:
            return sin(pi * (l1 - l2) / 3.0)**2
        else:
            return (6.0 + 10.0 * cos(2.0 * pi * (l1 - l2) / 3.0)) / 4.0
    elif lat == 2:  # Beryllium
        return 1.0 + cos(2.0 * pi * (2 * l1 + 4 * l2 + 3 * l3) / 6.0)
    elif lat == 3:  # BeO
        return ((1.0 + cos(2.0 * pi * (2 * l1 + 4 * l2 + 3 * l3) / 6.0)) *
                (7.54 + 4.24 + 11.31 * cos(3.0 * pi * l3 / 4.0)))
    elif lat == 4 or lat == 5:  # FCC
        e1 = 2.0 * pi * l1
        e2 = 2.0 * pi * (l1 + l2)
        e3 = 2.0 * pi * (l1 + l3)
        return (1.0 + cos(e1) + cos(e2) + cos(e3))**2 + (sin(e1) + sin(e2) + sin(e3))**2
    elif lat == 6:  # BCC
        e1 = 2.0 * pi * (l1 + l2 + l3)
        return (1.0 + cos(e1))**2 + sin(e1)**2
    return 0.0


# ============================================================================
# Skold approximation
# ============================================================================

def skold_approx(ssm, alpha, beta, nalpha, nbeta, itemp, ntempr, lat,
                 arat, awr, tev, ska, nka, dka, cfrac):
    """Apply Skold approximation for intermolecular coherence."""
    angst = 1.0e-8
    sc = 1.0
    if lat == 1:
        sc = THERM / tev
    amass = awr * AMASSN * AMU

    for i in range(nbeta):
        scoh_arr = np.zeros(nalpha)
        for j in range(nalpha):
            al = alpha[j] * sc / arat
            waven = angst * sqrt(2.0 * amass * tev * EV * al) / HBAR
            sk = terpk(ska, nka, dka, waven)
            ap = alpha[j] / sk

            # Find interpolation bracket
            kk = 0
            for k in range(nalpha):
                kk = k
                if ap < alpha[k]:
                    break
            if kk == 0:
                kk = 1

            if (ssm[i, kk - 1, itemp] == 0.0 or ssm[i, kk, itemp] == 0.0):
                scoh_arr[j] = 0.0
            else:
                # Log-log interpolation (terp1 with flag 5)
                x1, y1 = alpha[kk - 1], ssm[i, kk - 1, itemp]
                x2, y2 = alpha[kk], ssm[i, kk, itemp]
                if y1 > 0 and y2 > 0 and x1 > 0 and x2 > 0 and ap > 0:
                    scoh_arr[j] = exp(log(y1) + (log(ap) - log(x1)) * (log(y2) - log(y1)) / (log(x2) - log(x1)))
                else:
                    scoh_arr[j] = 0.0
            scoh_arr[j] *= sk

        for j in range(nalpha):
            ssm[i, j, itemp] = (1.0 - cfrac) * ssm[i, j, itemp] + cfrac * scoh_arr[j]


# ============================================================================
# ENDF output using endf-parserpy
# ============================================================================

def sigfig(x, ndig, idig):
    """Round x to ndig significant figures (matching Fortran sigfig exactly).

    Replicates NJOY's util.f90 sigfig, including its rounding bias and
    multiplicative bias to match Fortran output bit-for-bit.
    """
    bias = 1.0000000000001
    if x == 0.0:
        return 0.0
    aa = np.log10(abs(x))
    ipwr = int(aa)
    if aa < 0:
        ipwr = ipwr - 1
    ipwr = ndig - 1 - ipwr
    # Guard against overflow for extremely small/large values
    if ipwr > 300:
        return 0.0
    if ipwr < -300:
        return x * bias
    # Fortran: ii=nint(x*ten**ipwr+ten**(ndig-11))
    # The ten**(ndig-11) bias pushes values just past 0.5 for tie-breaking
    scaled = x * 10.0**ipwr + 10.0**(ndig - 11)
    ii = int(np.round(scaled))
    if abs(ii) >= 10**ndig:
        ii = ii // 10
        ipwr = ipwr - 1
    ii = ii + idig
    xx = ii * 10.0**(-ipwr)
    return xx * bias


def write_endf_output(filename, mat, za, awr, spr, npr, iel, ncold, nss,
                      b7, aws, sps, mss, nalpha, nbeta, lat,
                      alpha, beta, ssm, ssp, tempr, ntempr,
                      dwpix, dwp1, tempf, tempf1, twt, tbeta,
                      bragg, nedge, isym, ilog, smin, iprint,
                      comments=None, crystal_info=None):
    """Write ENDF-6 output file using endf-parserpy."""
    from endf_parserpy import EndfParserPy

    small = 1.0e-9

    # Compute bound scattering cross sections
    sb = spr * ((1.0 + awr) / awr)**2
    if aws != 0.0:
        sbs = sps * ((1.0 + aws) / aws)**2

    # For mixed moderators, merge ssm results
    if nss != 0 and b7 <= 0.0:
        srat = sbs / sb
        # ssm_principal was saved separately, need to merge
        # This is handled in the main loop
        pass

    # Compute Debye-Waller integral for ENDF output
    dwpix_out = dwpix.copy()
    dwp1_out = dwp1.copy()
    for i in range(ntempr):
        if nss == 0 or b7 > 0.0:
            dwpix_out[i] = dwpix[i] / (awr * tempr[i] * BK)
        else:
            dwpix_out[i] = dwpix[i] / (aws * tempr[i] * BK)
            dwp1_out[i] = dwp1[i] / (awr * tempr[i] * BK)

    # Determine symmetry type
    # isym: 0 = symmetric S, 1 = S for +/- beta (coldh),
    #        2 = ss for -beta, 3 = ss for +/- beta

    # Build ENDF dictionary
    parser = EndfParserPy()

    # Read a reference file to understand the complete structure, then build from scratch
    endf_dict = {}

    # ---- MF1/MT451 ----
    mf1 = {}
    mf1['MAT'] = mat
    mf1['MF'] = 1
    mf1['MT'] = 451
    mf1['ZA'] = za
    mf1['AWR'] = awr
    mf1['LRP'] = -1
    mf1['LFI'] = 0
    mf1['NLIB'] = 0
    mf1['NMOD'] = 0
    mf1['ELIS'] = 0.0
    mf1['STA'] = 0.0
    mf1['LIS'] = 0
    mf1['LISO'] = 0
    mf1['NFOR'] = 6
    mf1['AWI'] = 1.0
    mf1['EMAX'] = 0.0
    mf1['LREL'] = 0
    mf1['NSUB'] = 12  # thermal scattering sub-library
    mf1['NVER'] = 6
    mf1['TEMP'] = 0.0
    mf1['LDRV'] = 0

    # Text fields (must be padded to exact ENDF field widths)
    mf1['ZSYMAM'] = ' ' * 11
    mf1['ALAB'] = ' ' * 11
    mf1['EDATE'] = ' ' * 10
    mf1['AUTH'] = ' ' * 33
    mf1['REF'] = ' ' * 21
    mf1['DDATE'] = ' ' * 10
    mf1['RDATE'] = ' ' * 10
    mf1['ENDATE'] = ' ' * 8
    mf1['HSUB'] = {1: ' ' * 66, 2: ' ' * 66, 3: ' ' * 66}
    mf1['NWD'] = 0
    mf1['DESCRIPTION'] = {}

    # Populate text fields from comment cards
    # ENDF MF1/MT451 text records: NWD total records, structured as:
    #   Record 1: ZSYMAM(11) + ALAB(11) + EDATE(10) + AUTH(33) = 65 chars
    #   Record 2: REF(21) + DDATE(10) + RDATE(10) + ENDATE(8) = 49 chars
    #   Records 3-5: HSUB[1-3], 66 chars each
    #   Records 6+: DESCRIPTION[1..NWD-5], 66 chars each
    if comments is None:
        comments = []
    clean_comments = []
    for c in comments:
        c = c.strip()
        if (c.startswith("'") and c.endswith("'")) or \
           (c.startswith('"') and c.endswith('"')):
            c = c[1:-1]
        clean_comments.append(c)

    # Card 1 → ZSYMAM + ALAB + EDATE + AUTH
    if len(clean_comments) > 0:
        line0 = clean_comments[0].ljust(66)[:66]
        mf1['ZSYMAM'] = line0[:11]
        mf1['ALAB'] = line0[11:22]
        mf1['EDATE'] = line0[22:32]
        mf1['AUTH'] = line0[33:66]
    # Card 2 → REF + DDATE + RDATE + ENDATE
    # endf_parserpy layout: {1}blank + REF{21} + DDATE{10} + {1}blank + RDATE{10} + {12}pad + ENDATE{8} + {3}pad
    # Positions: [0]=blank, [1:22]=REF, [22:32]=DDATE, [32]=blank, [33:43]=RDATE, [43:55]=pad, [55:63]=ENDATE
    if len(clean_comments) > 1:
        line1 = clean_comments[1].ljust(66)[:66]
        mf1['REF'] = line1[1:22]
        mf1['DDATE'] = line1[22:32]
        mf1['RDATE'] = line1[33:43]
        mf1['ENDATE'] = line1[55:63]
    # Cards 3-5 → HSUB[1-3]
    for i in range(3):
        idx = i + 2
        if idx < len(clean_comments):
            mf1['HSUB'][i + 1] = clean_comments[idx].ljust(66)[:66]
    # Cards 6+ → DESCRIPTION[1..]
    desc_start = 5
    desc_lines = clean_comments[desc_start:] if len(clean_comments) > desc_start else []
    # NWD = total text records (5 header + description)
    mf1['NWD'] = len(clean_comments)
    mf1['DESCRIPTION'] = {i + 1: desc_lines[i].ljust(66)[:66]
                          for i in range(len(desc_lines))}

    # Directory - entries for MF1/MT451 + MF7/MT2 (if present) + MF7/MT4
    # NCx computed using NJOY's exact formulas from leapr.f90 lines 3126-3150
    nxc_sections = 1  # MF7/MT4 always present
    if iel != 0:
        nxc_sections += 1  # MF7/MT2
    nxc = nxc_sections + 1  # +1 for MF1/MT451 itself
    mf1['NXC'] = nxc
    mf1['MFx'] = {}
    mf1['MTx'] = {}
    mf1['NCx'] = {}
    mf1['MOD'] = {}

    # Compute NCx for MF1/MT451: 4 header CONTs + NWD text + NXC directory
    nc_mt451 = 4 + mf1['NWD'] + nxc

    # Compute NCx for MF7/MT4 (NJOY formula: leapr.f90 lines 3147-3149)
    per_beta = 2 + (2 * nalpha + 4) // 6
    if ntempr > 1:
        per_beta += (ntempr - 1) * (1 + (nalpha + 5) // 6)
    nc_mt4 = 5 + nbeta * per_beta

    # Compute NCx for MF7/MT2 if present (leapr.f90 lines 3135-3138)
    nc_mt2 = 0
    if iel < 0:
        nc_mt2 = 3 + (2 * ntempr + 4) // 6
    elif iel > 0:
        nc_mt2 = 3 + (2 * nedge + 4) // 6
        if ntempr > 1:
            nc_mt2 += (ntempr - 1) * (1 + (nedge + 5) // 6)

    idx = 1
    mf1['MFx'][idx] = 1
    mf1['MTx'][idx] = 451
    mf1['NCx'][idx] = nc_mt451
    mf1['MOD'][idx] = 0
    idx += 1
    if iel != 0:
        mf1['MFx'][idx] = 7
        mf1['MTx'][idx] = 2
        mf1['NCx'][idx] = nc_mt2
        mf1['MOD'][idx] = 0
        idx += 1
    mf1['MFx'][idx] = 7
    mf1['MTx'][idx] = 4
    mf1['NCx'][idx] = nc_mt4
    mf1['MOD'][idx] = 0

    # ---- MF7/MT4 (inelastic) ----
    mf7mt4 = {}
    mf7mt4['MAT'] = mat
    mf7mt4['MF'] = 7
    mf7mt4['MT'] = 4
    mf7mt4['ZA'] = za
    mf7mt4['AWR'] = awr
    mf7mt4['LAT'] = lat
    mf7mt4['LASYM'] = isym
    mf7mt4['LLN'] = 1 if ilog != 0 else 0
    mf7mt4['NS'] = nss if nss > 0 else (1 if nss == 0 and mss == 0 else nss)

    # Compute NS properly: NS is the number of non-principal scatterers
    # But for the B array, we need it to determine NI
    ns_val = 0
    if nss > 0:
        ns_val = nss
    mf7mt4['NS'] = ns_val

    ni = 6
    if ns_val > 0:
        ni = 6 * (ns_val + 1)
    mf7mt4['NI'] = ni

    # B array
    mf7mt4['B'] = {}
    mf7mt4['B'][1] = sigfig(spr * npr * beta[nbeta - 1], 7, 0)  # epsilon
    mf7mt4['B'][2] = beta[nbeta - 1]
    mf7mt4['B'][3] = awr
    mf7mt4['B'][4] = sigfig(THERM * beta[nbeta - 1], 7, 0)
    mf7mt4['B'][5] = 0.0
    mf7mt4['B'][6] = float(npr)

    if ns_val > 0:
        mf7mt4['B'][7] = b7
        mf7mt4['B'][8] = mss * sps
        mf7mt4['B'][9] = aws
        mf7mt4['B'][10] = 0.0
        mf7mt4['B'][11] = 0.0
        mf7mt4['B'][12] = float(mss)

    # Wait - let me match the reference format exactly.
    # From the reference tape: B[1]=40.898 = NPR*SPR = 2*20.449
    # B[2]=25.0 = beta_max (energy transfer limit)
    # B[3]=0.99917 = AWR
    # B[4]=0.6325 = SPR/NPR... no, that's sigma_free per atom?
    # Actually looking at the Fortran endout:
    #   scr(7)=npr*spr    -> B[1] = NPR*SPR (bound? free?)
    # Wait no: scr(7)=npr*spr -> this is the total free-atom cross section
    # Let me re-read endout more carefully

    # From Fortran endout (lines 3309-3312):
    # scr(7)=npr*spr           -> B[1] = principal: sigma * natom
    # scr(8)=beta(nbeta)        -> B[2] = beta_max
    # scr(9)=awr                -> B[3] = AWR
    # scr(10)=sigfig(therm*beta(nbeta),7,0) -> B[4] = E_max = 0.0253*beta_max
    # scr(11)=0                 -> B[5] = 0
    # scr(12)=npr               -> B[6] = natom
    # For secondary (lines 3316-3321):
    # scr(13)=b7                -> B[7] = type
    # scr(14)=mss*sps           -> B[8] = secondary: sigma * natom
    # scr(15)=aws               -> B[9] = secondary AWR
    # scr(16)=0                 -> B[10] = 0
    # scr(17)=0                 -> B[11] = 0
    # scr(18)=mss               -> B[12] = secondary natom

    mf7mt4['B'][1] = npr * spr
    mf7mt4['B'][2] = beta[nbeta - 1]
    mf7mt4['B'][3] = awr
    mf7mt4['B'][4] = sigfig(THERM * beta[nbeta - 1], 7, 0)
    mf7mt4['B'][5] = 0.0
    mf7mt4['B'][6] = float(npr)

    if ns_val > 0:
        mf7mt4['B'][7] = b7
        mf7mt4['B'][8] = mss * sps
        mf7mt4['B'][9] = aws
        mf7mt4['B'][10] = 0.0
        mf7mt4['B'][11] = 0.0
        mf7mt4['B'][12] = float(mss)

    # Beta grid
    nbt = nbeta
    if isym == 1 or isym == 3:
        nbt = 2 * nbeta - 1
    mf7mt4['NB'] = nbt
    mf7mt4['beta_interp'] = {'NBT': [nbt], 'INT': [4]}

    mf7mt4['beta'] = {}
    mf7mt4['LT'] = {}
    mf7mt4['S_table'] = {}
    mf7mt4['T0'] = tempr[0]

    if ntempr > 1:
        mf7mt4['T'] = {j: tempr[j] for j in range(1, ntempr)}
        mf7mt4['LI'] = {j: 4 for j in range(1, ntempr)}
        mf7mt4['NP'] = nalpha
        # S[q][i][j]: q=alpha(1..NP), i=beta(1..NB), j=temp(1..LT)
        mf7mt4['S'] = {q: {} for q in range(1, nalpha + 1)}

    sc_vals = np.ones(ntempr)
    for nt in range(ntempr):
        if lat == 1:
            sc_vals[nt] = THERM / (BK * tempr[nt])

    for ii in range(1, nbt + 1):
        # Determine beta value
        if isym % 2 == 0:
            i_beta = ii - 1  # 0-based index into beta array
            beta_val = beta[i_beta]
        else:
            if ii < nbeta:
                i_beta = nbeta - ii  # reversed
                beta_val = -beta[i_beta]
            else:
                i_beta = ii - nbeta  # 0-based
                beta_val = beta[i_beta]

        mf7mt4['beta'][ii] = beta_val
        mf7mt4['LT'][ii] = ntempr - 1

        # S values for T0 (first temperature)
        nt = 0
        be = beta_val * sc_vals[nt]
        s_values = np.zeros(nalpha)

        for j in range(nalpha):
            s_val = _compute_endf_s(ssm, ssp, isym, ilog, ii, nbeta, j, nt,
                                     be, smin, small)
            s_values[j] = s_val

        mf7mt4['S_table'][ii] = {
            'NBT': [nalpha],
            'INT': [4],
            'alpha': alpha.tolist(),
            'S': s_values.tolist()
        }

        # Additional temperatures: S[q][i][j] = S[alpha][beta][temp]
        if ntempr > 1:
            for j_temp in range(1, ntempr):
                be = beta_val * sc_vals[j_temp]
                for q in range(nalpha):
                    s_val = _compute_endf_s(ssm, ssp, isym, ilog, ii, nbeta,
                                            q, j_temp, be, smin, small)
                    if ii not in mf7mt4['S'][q + 1]:
                        mf7mt4['S'][q + 1][ii] = {}
                    mf7mt4['S'][q + 1][ii][j_temp] = s_val

    # Effective temperature table(s)
    # Fortran NJOY writes tempf1 (principal) to the first TAB1 (Teff1 slot),
    # and tempf (secondary) to the second TAB1 (Teff0 slot).
    # Match this ordering for bit-identical output.
    if nss != 0 and b7 <= 0.0:
        mf7mt4['teff1_table'] = {
            'NBT': [ntempr],
            'INT': [2],
            'Tint': [sigfig(tempr[i], 7, 0) for i in range(ntempr)],
            'Teff1': [sigfig(tempf[i], 7, 0) for i in range(ntempr)]
        }

    mf7mt4['teff0_table'] = {
        'NBT': [ntempr],
        'INT': [2],
        'Tint': [sigfig(tempr[i], 7, 0) for i in range(ntempr)],
        'Teff0': [sigfig(tempf1[i], 7, 0) for i in range(ntempr)]
    }

    # ---- MF7/MT2 (elastic) ----
    mf7mt2 = None
    if iel == 10 and crystal_info is not None:
        # Generalized elastic output (CEF or MEF)
        mf7mt2 = _build_generalized_elastic(
            mat, za, awr, bragg, nedge, ntempr, tempr,
            crystal_info, dwpix_out, sb)

    elif iel < 0:
        # Incoherent elastic (LTHR=2)
        mf7mt2 = {}
        mf7mt2['MAT'] = mat
        mf7mt2['MF'] = 7
        mf7mt2['MT'] = 2
        mf7mt2['ZA'] = za
        mf7mt2['AWR'] = awr
        mf7mt2['LTHR'] = 2  # incoherent elastic

        ndw = max(ntempr, 2)
        mf7mt2['SB'] = sb * npr
        mf7mt2['NBT'] = [ndw]
        mf7mt2['INT'] = [2]  # linear-linear interpolation
        mf7mt2['Tint'] = []
        mf7mt2['Wp'] = []

        for i in range(ndw):
            idx_t = min(i, ntempr - 1)
            mf7mt2['Tint'].append(tempr[idx_t])
            mf7mt2['Wp'].append(sigfig(dwpix_out[idx_t], 7, 0))

    elif iel >= 1:
        # Coherent elastic (standard hardcoded materials)
        mf7mt2 = _build_coherent_elastic(mat, za, awr, bragg, nedge, ntempr,
                                          tempr, dwpix_out, dwp1_out, nss, b7)

    # ---- Assemble and write ----
    endf_dict = {}
    endf_dict[0] = {0: {'MAT': 1, 'MF': 0, 'MT': 0,
                         'TAPEDESCR': ' ' * 66}}
    endf_dict[1] = {451: mf1}
    endf_dict[7] = {}
    if mf7mt2 is not None:
        endf_dict[7][2] = mf7mt2
    endf_dict[7][4] = mf7mt4

    parser.writefile(filename, endf_dict, overwrite=True)

    # Post-process: fix SEND/FEND/MEND/TEND record formatting
    # endf_parserpy writes "0.000000+0 0.000000+0 ..." but NJOY writes blank fields
    with open(filename, 'r') as f:
        lines = f.readlines()
    with open(filename, 'w') as f:
        for line in lines:
            if len(line) >= 75:
                # Check for SEND (MT=99999), FEND (MF=0), MEND (MAT=0), TEND (MAT=-1)
                try:
                    mt_field = int(line[72:75])
                except (ValueError, IndexError):
                    mt_field = None
                try:
                    mat_field = int(line[66:70])
                except (ValueError, IndexError):
                    mat_field = None
                is_special = False
                if mt_field is not None and mt_field == 0:
                    is_special = True  # FEND or MEND or TEND
                if line[70:75].strip() == '99999':
                    is_special = True  # SEND
                if is_special:
                    # Blank out the data fields (first 66 chars), keep MAT/MF/MT/line#
                    line = ' ' * 66 + line[66:]
            f.write(line)
        # Ensure file ends with a newline (Fortran NJOY always does)
        if lines and not lines[-1].endswith('\n'):
            f.write('\n')
    print(f"ENDF output written to {filename}")


def _compute_endf_s(ssm, ssp, isym, ilog, ii, nbeta, j, nt, be, smin, small):
    """Compute the ENDF S value for a given (beta, alpha, temperature) point."""
    tiny = -999.0

    if isym == 0:
        # Symmetric S(a,b): store S*exp(-b/2)
        i_beta = ii - 1  # 0-based
        if ilog == 0:
            s_val = ssm[i_beta, j, nt] * exp(-be / 2.0)
            if s_val >= small:
                s_val = sigfig(s_val, 7, 0)
            else:
                s_val = sigfig(s_val, 6, 0)
        else:
            if ssm[i_beta, j, nt] > 0.0:
                s_val = log(ssm[i_beta, j, nt]) - be / 2.0
                s_val = sigfig(s_val, 7, 0)
            else:
                s_val = tiny

    elif isym == 1:
        # Asymmetric S for +/- beta (cold hydrogen)
        if ii < nbeta:
            i_beta = nbeta - ii  # 0-based
            if ilog == 0:
                s_val = ssm[i_beta, j, nt] * exp(be / 2.0)
                if s_val >= small:
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = sigfig(s_val, 6, 0)
            else:
                if ssm[i_beta, j, nt] > 0.0:
                    s_val = log(ssm[i_beta, j, nt]) + be / 2.0
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = tiny
        else:
            i_beta = ii - nbeta  # 0-based
            if ilog == 0:
                s_val = ssp[i_beta, j, nt] * exp(be / 2.0)
                if s_val >= small:
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = sigfig(s_val, 6, 0)
            else:
                if ssp[i_beta, j, nt] > 0.0:
                    s_val = log(ssp[i_beta, j, nt]) + be / 2.0
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = tiny

    elif isym == 2:
        # Asymmetric SS for -beta (isabt=1)
        i_beta = ii - 1
        if ilog == 0:
            s_val = ssm[i_beta, j, nt]
            if s_val >= small:
                s_val = sigfig(s_val, 7, 0)
            else:
                s_val = sigfig(s_val, 6, 0)
        else:
            if ssm[i_beta, j, nt] > 0.0:
                s_val = log(ssm[i_beta, j, nt])
                s_val = sigfig(s_val, 7, 0)
            else:
                s_val = tiny

    elif isym == 3:
        # Asymmetric SS for +/- beta
        if ii < nbeta:
            i_beta = nbeta - ii
            if ilog == 0:
                s_val = ssm[i_beta, j, nt]
                if s_val >= small:
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = sigfig(s_val, 6, 0)
            else:
                if ssm[i_beta, j, nt] > 0.0:
                    s_val = log(ssm[i_beta, j, nt])
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = tiny
        else:
            i_beta = ii - nbeta
            if ilog == 0:
                s_val = ssp[i_beta, j, nt]
                if s_val >= small:
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = sigfig(s_val, 6, 0)
            else:
                if ssp[i_beta, j, nt] > 0.0:
                    s_val = log(ssp[i_beta, j, nt])
                    s_val = sigfig(s_val, 7, 0)
                else:
                    s_val = tiny
    else:
        s_val = 0.0

    if ilog == 0 and s_val < smin:
        s_val = 0.0

    return s_val


def _build_coherent_elastic(mat, za, awr, bragg, nedge, ntempr,
                             tempr, dwpix, dwp1, nss, b7):
    """Build MF7/MT2 coherent elastic section."""
    tol = 0.9e-7

    mf7mt2 = {}
    mf7mt2['MAT'] = mat
    mf7mt2['MF'] = 7
    mf7mt2['MT'] = 2
    mf7mt2['ZA'] = za
    mf7mt2['AWR'] = awr
    mf7mt2['LTHR'] = 1  # coherent elastic
    mf7mt2['T0'] = tempr[0]
    mf7mt2['LT'] = ntempr - 1

    # Thin out the 1/e part at high energies
    w0 = dwpix[0]
    if nss > 0 and b7 == 0.0:
        w0 = (dwpix[0] + dwp1[0]) / 2.0

    total_sum = 0.0
    suml = 0.0
    jmax = 0
    for j in range(nedge):
        e = bragg[j][0]
        total_sum += exp(-4.0 * w0 * e) * bragg[j][1]
        if total_sum - suml > tol * total_sum:
            jmax = j + 1
            suml = total_sum

    # First temperature: TAB1 with energies and cumulative S
    mf7mt2['NP'] = jmax
    mf7mt2['S_T0_table'] = {
        'NBT': [jmax],
        'INT': [1],  # histogram interpolation
    }

    energies = []
    s_cumulative = []
    s = 0.0
    for j in range(nedge):
        e = bragg[j][0]
        s += exp(-4.0 * w0 * e) * bragg[j][1]
        if j < jmax:
            energies.append(sigfig(e, 7, 0))
            s_cumulative.append(sigfig(s, 7, 0))
        else:
            # Overwrite last entry (matches Fortran endout loop behavior)
            energies[-1] = sigfig(e, 7, 0)
            s_cumulative[-1] = sigfig(s, 7, 0)

    mf7mt2['S_T0_table']['Eint'] = energies
    mf7mt2['S_T0_table']['S'] = s_cumulative

    # Additional temperatures
    if ntempr > 1:
        mf7mt2['T'] = {}
        mf7mt2['LI'] = 2  # lin-lin interpolation (scalar, not per-temp)
        # S[q][i] where q=point index (1..NP), i=temp index (1..LT)
        mf7mt2['S'] = {q: {} for q in range(1, jmax + 1)}

        for i in range(1, ntempr):
            mf7mt2['T'][i] = tempr[i]

            w = dwpix[i]
            if nss > 0 and b7 == 0.0:
                w = (dwpix[i] + dwp1[i]) / 2.0

            s = 0.0
            jj = 0
            for j in range(nedge):
                if j < jmax:
                    jj = j
                # For j >= jmax, jj stays at jmax-1 (Fortran behavior)
                e_sf = sigfig(bragg[jj][0], 7, 0)
                s += exp(-4.0 * w * e_sf) * bragg[jj][1]
                mf7mt2['S'][jj + 1][i] = sigfig(s, 7, 0)

    return mf7mt2


# ============================================================================
# Generalized elastic (iel=10) support functions
# ============================================================================

def _compute_per_species_msd(crystal_info, tempr_arr, ntempr, dwpix):
    """Compute per-species MSD (Debye-Waller lambda) from partial phonon spectra.

    For each atom type that has a matching partial phonon spectrum, compute
    the DW lambda (f0) at each temperature using the same method as LEAPR's
    start() function.  Atom types without a matching spectrum fall back to
    LEAPR's dwpix (principal scatterer DW).

    Results are stored in crystal_info['atom_types'][i]['dwpix'][itemp].
    """
    atom_types = crystal_info['atom_types']
    partial_spectra = crystal_info['partial_spectra']

    for at in atom_types:
        at['dwpix'] = np.zeros(ntempr)

    for iat, at in enumerate(atom_types):
        sp_idx = at['spectrum_idx']
        if sp_idx is None:
            # No partial spectrum — fall back to LEAPR's dwpix
            at['dwpix'][:] = dwpix[:ntempr]
            print(f"    Atom type {iat+1} (Z={at['Z']}, A={at['A']}): "
                  f"no partial spectrum, using principal DW")
            continue

        sp = partial_spectra[sp_idx]
        rho = sp['rho']
        delta_e = sp['delta']     # energy spacing in eV
        ni = sp['ni']

        for itemp in range(ntempr):
            temp = tempr_arr[itemp]
            tev = BK * temp
            deltab = delta_e / tev  # energy grid in units of kT

            # Normalize the spectrum: ∫ ρ(ε)/ε dε should give tbeta
            # Same approach as LEAPR's start(): transform p, normalize, then fsum(0)
            p = np.copy(rho[:ni].astype(float))

            # Transform: p[j] = rho[j] / (beta * 2sinh(beta/2))
            u = deltab
            v = exp(deltab / 2.0)
            # Handle the j=0 case (limiting value)
            if ni > 1:
                p[0] = p[1] / deltab**2
            vv = v
            for j in range(1, ni):
                denom = u * (vv - 1.0 / vv)
                if abs(denom) > 1e-30:
                    p[j] = p[j] / denom
                else:
                    p[j] = 0.0
                vv = v * vv
                u += deltab

            # Normalize: tbeta = weight of continuous spectrum
            # For partial spectra, assume full weight (tbeta=1)
            tau = 0.5
            an = fsum(1, p, ni, tau, deltab)
            if an > 0:
                for i in range(ni):
                    p[i] = p[i] / an

            # DW lambda = fsum(0, ...)
            f0 = fsum(0, p, ni, tau, deltab)
            at['dwpix'][itemp] = f0

        print(f"    Atom type {iat+1} (Z={at['Z']}, A={at['A']}): "
              f"DW from partial spectrum, f0[0]={at['dwpix'][0]:.6f}")


def _build_generalized_elastic(mat, za, awr, bragg, nedge, ntempr, tempr,
                                crystal_info, dwpix_out, sb):
    """Build MF7/MT2 for generalized elastic (iel=10).

    Implements both CEF (elastic_mode=1) and MEF (elastic_mode=2) following
    the paper: K. Ramic, J. I. Damian Marquez, et al., NIM-A 1027 (2022) 166227.

    For CEF:
      - Single atom (nat=1): Eq 24/25 — store dominant channel, scale by
        (σ_coh + σ_inc) / σ_dominant
      - Polyatomic (nat>1): Eq 26 — DC atom gets coherent elastic scaled
        by 1/f_DC; other atoms get incoherent elastic with redistribution

    For MEF:
      - Eq 23: LTHR=3 with coherent elastic (per atom) + incoherent elastic
    """
    tol = 0.9e-7

    elastic_mode = crystal_info['elastic_mode']
    atom_types = crystal_info['atom_types']
    nat = crystal_info['nat']
    principal_idx = crystal_info['principal_atom_idx']
    dc_idx = crystal_info['dc_atom_idx']  # None if MEF or nat==1
    principal_at = atom_types[principal_idx]

    sigma_coh_p = principal_at['sigma_coh']   # barns
    sigma_inc_p = principal_at['sigma_inc']   # barns
    f_principal = principal_at['fraction']

    # ---- MEF (elastic_mode=2): LTHR=3 ----
    if elastic_mode == 2:
        return _build_mef_elastic(mat, za, awr, bragg, nedge, ntempr, tempr,
                                   crystal_info, dwpix_out)

    # ---- CEF (elastic_mode=1) ----

    if nat == 1:
        # Single atom CEF: Eq 24 or 25
        if sigma_coh_p > sigma_inc_p:
            # Coherent approximation (Eq 24)
            # Scale Bragg edges by (σ_coh + σ_inc) / σ_coh
            scale = (sigma_coh_p + sigma_inc_p) / sigma_coh_p
            print(f"  CEF single atom: coherent approx, "
                  f"scale={(sigma_coh_p + sigma_inc_p):.4f}/{sigma_coh_p:.4f} = {scale:.4f}")
            return _build_cef_coherent(mat, za, awr, bragg, nedge, ntempr,
                                        tempr, dwpix_out, scale,
                                        crystal_info=crystal_info)
        else:
            # Incoherent approximation (Eq 25)
            # Scale SB by (σ_coh + σ_inc) / σ_inc
            scale = (sigma_coh_p + sigma_inc_p) / sigma_inc_p
            print(f"  CEF single atom: incoherent approx, "
                  f"SB scale={scale:.4f}")
            return _build_cef_incoherent(mat, za, awr, ntempr, tempr,
                                          dwpix_out, sigma_inc_p * scale)

    # Polyatomic CEF (nat > 1): Eq 26
    if principal_idx == dc_idx:
        # Principal scatterer IS the DC atom → LTHR=1 (coherent elastic)
        # Scale by 1/f_DC to get per-DC-atom cross section
        f_dc = principal_at['fraction']
        scale = 1.0 / f_dc
        print(f"  CEF polyatomic: principal is DC atom, "
              f"scale=1/f_DC=1/{f_dc:.4f}={scale:.4f}")
        return _build_cef_coherent(mat, za, awr, bragg, nedge, ntempr,
                                    tempr, dwpix_out, scale,
                                    crystal_info=crystal_info)
    else:
        # Principal scatterer is NOT the DC atom → LTHR=2 (incoherent elastic)
        # with redistribution factor from Eq 26
        dc_at = atom_types[dc_idx]
        f_dc = dc_at['fraction']
        sigma_inc_dc = dc_at['sigma_inc']

        # Redistribution factor: [1 + f_DC/(1-f_DC) × σ_inc_DC/σ_inc_i]
        redist = 1.0 + (f_dc / (1.0 - f_dc)) * (sigma_inc_dc / sigma_inc_p)

        # SB for incoherent elastic: σ_inc of this atom × redistribution factor
        sb_redist = sigma_inc_p * redist
        print(f"  CEF polyatomic: principal is NOT DC atom, "
              f"redist factor={redist:.4f}, SB={sb_redist:.4f} b")
        return _build_cef_incoherent(mat, za, awr, ntempr, tempr,
                                      dwpix_out, sb_redist)


def _build_cef_coherent(mat, za, awr, bragg, nedge, ntempr, tempr,
                         dwpix_out, scale, crystal_info=None):
    """Build LTHR=1 (coherent elastic) section with a multiplicative scale.

    The scale factor accounts for:
    - Single atom CEF: (σ_coh + σ_inc)/σ_coh  (Eq 24)
    - Polyatomic CEF DC atom: 1/f_DC  (Eq 26)

    If crystal_info with species_corr is provided, per-species Debye-Waller
    factors are applied inside the structure factor (matching NCrystal):
        δ_j = scale × Σ_{s,t} b_s b_t exp(-2(W_s+W_t) E_j) D_{st,j}
    where W_s = dwpix_s / (awr_s T kB) is the per-species ENDF DW parameter.
    """
    tol = 0.9e-7

    # --- Per-species DW setup ---
    use_ps = (crystal_info is not None and
              'species_corr' in crystal_info and
              crystal_info['species_corr'] is not None)

    if use_ps:
        atom_types = crystal_info['atom_types']
        sc = crystal_info['species_corr']   # (nedge+1, nsp, nsp)
        nsp = len(atom_types)
        # b_coh in sqrt(barn) = b_coh_fm / 10
        b_sqb = [at['b_coh'] / 10.0 for at in atom_types]
        # W_s[si][itemp] = dwpix_s / (awr_s × T × kB)  [1/eV]
        W_ps = [[at['dwpix'][it] / (at['awr'] * tempr[it] * BK)
                 for it in range(ntempr)] for at in atom_types]

        print(f"    Per-species DW (ENDF W, 1/eV) at T={tempr[0]:.2f}K:")
        for si, at in enumerate(atom_types):
            print(f"      {at['Z']}-{at['A']}: W={W_ps[si][0]:.6f}")

    def _edge_delta(j, itemp, energy=None):
        """DW-weighted edge contribution at temperature itemp."""
        e = energy if energy is not None else bragg[j][0]
        if use_ps:
            delta = 0.0
            for si in range(nsp):
                w_si = W_ps[si][itemp]
                for ti in range(nsp):
                    dw = exp(-2.0 * (w_si + W_ps[ti][itemp]) * e)
                    delta += b_sqb[si] * b_sqb[ti] * dw * sc[j, si, ti]
            return delta * scale
        else:
            return exp(-4.0 * dwpix_out[itemp] * e) * bragg[j][1] * scale

    # --- Build ENDF dict ---
    mf7mt2 = {}
    mf7mt2['MAT'] = mat
    mf7mt2['MF'] = 7
    mf7mt2['MT'] = 2
    mf7mt2['ZA'] = za
    mf7mt2['AWR'] = awr
    mf7mt2['LTHR'] = 1  # coherent elastic
    mf7mt2['T0'] = tempr[0]
    mf7mt2['LT'] = ntempr - 1

    # Thin out negligible edges at first temperature and cache deltas
    deltas_T0 = []
    total_sum = 0.0
    suml = 0.0
    jmax = 0
    for j in range(nedge):
        d = _edge_delta(j, 0)
        deltas_T0.append(d)
        total_sum += d
        if total_sum - suml > tol * total_sum:
            jmax = j + 1
            suml = total_sum

    # First temperature: TAB1 with energies and cumulative S
    mf7mt2['NP'] = jmax
    mf7mt2['S_T0_table'] = {
        'NBT': [jmax],
        'INT': [1],  # histogram interpolation
    }

    energies = []
    s_cumulative = []
    s = 0.0
    for j in range(nedge):
        s += deltas_T0[j]
        if j < jmax:
            energies.append(sigfig(bragg[j][0], 7, 0))
            s_cumulative.append(sigfig(s, 7, 0))
        else:
            energies[-1] = sigfig(bragg[j][0], 7, 0)
            s_cumulative[-1] = sigfig(s, 7, 0)

    mf7mt2['S_T0_table']['Eint'] = energies
    mf7mt2['S_T0_table']['S'] = s_cumulative

    # Additional temperatures
    if ntempr > 1:
        mf7mt2['T'] = {}
        mf7mt2['LI'] = 2  # lin-lin interpolation
        mf7mt2['S'] = {q: {} for q in range(1, jmax + 1)}

        for i in range(1, ntempr):
            mf7mt2['T'][i] = tempr[i]

            s = 0.0
            jj = 0
            for j in range(nedge):
                if j < jmax:
                    jj = j
                e_sf = sigfig(bragg[jj][0], 7, 0)
                s += _edge_delta(jj, i, energy=e_sf)
                mf7mt2['S'][jj + 1][i] = sigfig(s, 7, 0)

    return mf7mt2


def _build_cef_incoherent(mat, za, awr, ntempr, tempr, dwpix_out, sb_value):
    """Build LTHR=2 (incoherent elastic) section for CEF non-DC atom.

    sb_value is the effective bound cross section (barns), which may include
    the redistribution factor from Eq 26.
    """
    mf7mt2 = {}
    mf7mt2['MAT'] = mat
    mf7mt2['MF'] = 7
    mf7mt2['MT'] = 2
    mf7mt2['ZA'] = za
    mf7mt2['AWR'] = awr
    mf7mt2['LTHR'] = 2  # incoherent elastic

    ndw = max(ntempr, 2)
    mf7mt2['SB'] = sb_value
    mf7mt2['NBT'] = [ndw]
    mf7mt2['INT'] = [2]  # linear-linear interpolation
    mf7mt2['Tint'] = []
    mf7mt2['Wp'] = []

    for i in range(ndw):
        idx_t = min(i, ntempr - 1)
        mf7mt2['Tint'].append(tempr[idx_t])
        mf7mt2['Wp'].append(sigfig(dwpix_out[idx_t], 7, 0))

    return mf7mt2


def _build_mef_elastic(mat, za, awr, bragg, nedge, ntempr, tempr,
                        crystal_info, dwpix_out):
    """Build MF7/MT2 with LTHR=3 (mixed elastic format).

    Stores both coherent elastic (Bragg edges, per atom) and incoherent
    elastic in a single section, following the proposed ENDF extension
    (Eq 23 from the paper).

    σ^el_i = σ^coh/N + σ^inc_i
    """
    tol = 0.9e-7
    principal_at = crystal_info['atom_types'][crystal_info['principal_atom_idx']]
    sigma_inc_p = principal_at['sigma_inc']  # barns

    mf7mt2 = {}
    mf7mt2['MAT'] = mat
    mf7mt2['MF'] = 7
    mf7mt2['MT'] = 2
    mf7mt2['ZA'] = za
    mf7mt2['AWR'] = awr
    mf7mt2['LTHR'] = 3  # mixed elastic

    # --- Per-species DW setup (same as CEF) ---
    use_ps = ('species_corr' in crystal_info and
              crystal_info['species_corr'] is not None)

    if use_ps:
        atom_types = crystal_info['atom_types']
        sc = crystal_info['species_corr']
        nsp = len(atom_types)
        b_sqb = [at['b_coh'] / 10.0 for at in atom_types]
        W_ps = [[at['dwpix'][it] / (at['awr'] * tempr[it] * BK)
                 for it in range(ntempr)] for at in atom_types]

    def _edge_delta(j, itemp, energy=None):
        e = energy if energy is not None else bragg[j][0]
        if use_ps:
            delta = 0.0
            for si in range(nsp):
                w_si = W_ps[si][itemp]
                for ti in range(nsp):
                    dw = exp(-2.0 * (w_si + W_ps[ti][itemp]) * e)
                    delta += b_sqb[si] * b_sqb[ti] * dw * sc[j, si, ti]
            return delta
        else:
            return exp(-4.0 * dwpix_out[itemp] * e) * bragg[j][1]

    # --- Coherent elastic part (Bragg edges, per-atom average) ---
    deltas_T0 = []
    total_sum = 0.0
    suml = 0.0
    jmax = 0
    for j in range(nedge):
        d = _edge_delta(j, 0)
        deltas_T0.append(d)
        total_sum += d
        if total_sum - suml > tol * total_sum:
            jmax = j + 1
            suml = total_sum

    mf7mt2['T0'] = tempr[0]
    mf7mt2['LT'] = ntempr - 1
    mf7mt2['NP'] = jmax

    mf7mt2['S_T0_table'] = {
        'NBT': [jmax],
        'INT': [1],
    }

    energies = []
    s_cumulative = []
    s = 0.0
    for j in range(nedge):
        s += deltas_T0[j]
        if j < jmax:
            energies.append(sigfig(bragg[j][0], 7, 0))
            s_cumulative.append(sigfig(s, 7, 0))
        else:
            energies[-1] = sigfig(bragg[j][0], 7, 0)
            s_cumulative[-1] = sigfig(s, 7, 0)

    mf7mt2['S_T0_table']['Eint'] = energies
    mf7mt2['S_T0_table']['S'] = s_cumulative

    if ntempr > 1:
        mf7mt2['T'] = {}
        mf7mt2['LI'] = 2
        mf7mt2['S'] = {q: {} for q in range(1, jmax + 1)}

        for i in range(1, ntempr):
            mf7mt2['T'][i] = tempr[i]

            s = 0.0
            jj = 0
            for j in range(nedge):
                if j < jmax:
                    jj = j
                e_sf = sigfig(bragg[jj][0], 7, 0)
                s += _edge_delta(jj, i, energy=e_sf)
                mf7mt2['S'][jj + 1][i] = sigfig(s, 7, 0)

    # --- Incoherent elastic part ---
    # Uses same key names as LTHR=2 (SB, NBT, INT, Tint, Wp)
    # endf-parserpy handles LTHR=3 natively with this layout
    ndw = max(ntempr, 2)
    mf7mt2['SB'] = sigma_inc_p
    mf7mt2['NBT'] = [ndw]
    mf7mt2['INT'] = [2]
    mf7mt2['Tint'] = []
    mf7mt2['Wp'] = []

    for i in range(ndw):
        idx_t = min(i, ntempr - 1)
        mf7mt2['Tint'].append(tempr[idx_t])
        mf7mt2['Wp'].append(sigfig(dwpix_out[idx_t], 7, 0))

    print(f"  MEF: LTHR=3, {jmax} Bragg edges, σ_inc={sigma_inc_p:.4f} b")

    return mf7mt2


# ============================================================================
# Main LEAPR driver
# ============================================================================

def run_leapr(input_file, output_file):
    """Run the LEAPR calculation from an input file."""

    # Parse input
    tokens, raw_lines, start_line = parse_leapr_input(input_file)
    reader = TokenReader(tokens)

    # Card 1: output unit (ignored, we write directly)
    nout = reader.read_ints(1)[0]

    # Card 2: title
    title = reader.read_string()
    print(f"LEAPR: {title}")

    # Card 3: run control
    vals = reader.read_ints(3, defaults=[1, 1, 100])
    ntempr, iprint, nphon = vals

    # Card 4: ENDF output control
    fvals = reader.read_floats(5, defaults=[0, 0, 0, 0, 1.0e-75])
    mat = int(fvals[0])
    za = fvals[1]
    isabt = int(fvals[2])
    ilog = int(fvals[3])
    smin = fvals[4]

    print(f"  ntempr={ntempr}, iprint={iprint}, nphon={nphon}")
    print(f"  mat={mat}, za={za}, isabt={isabt}, ilog={ilog}")

    # Card 5: principal scatterer control
    fvals = reader.read_floats(6, defaults=[0, 0, 0, 0, 0, 0])
    awr = fvals[0]
    spr = fvals[1]
    npr = int(fvals[2])
    iel = int(fvals[3])
    ncold = int(fvals[4])
    nsk = int(fvals[5])

    print(f"  awr={awr}, spr={spr}, npr={npr}, iel={iel}, ncold={ncold}, nsk={nsk}")

    # Card 6: secondary scatterer control
    fvals = reader.read_floats(5, defaults=[0, 0, 0, 0, 0])
    nss = int(fvals[0])
    b7 = fvals[1]
    aws = fvals[2]
    sps = fvals[3]
    mss = int(fvals[4])

    if nss > 0:
        print(f"  nss={nss}, b7={b7}, aws={aws}, sps={sps}, mss={mss}")

    # ---- Generalized coherent elastic cards (iel=10) ----
    crystal_info = None  # will hold parsed crystal/elastic data if iel==10

    if iel == 10:
        # Card 6b: elastic_mode, nat, nspec
        fvals = reader.read_floats(3, defaults=[0, 0, 0])
        elastic_mode = int(fvals[0])   # 1=CEF, 2=MEF
        nat = int(fvals[1])            # number of distinct atom types
        nspec = int(fvals[2])          # number of partial phonon spectra

        if elastic_mode not in (1, 2):
            raise ValueError(f"elastic_mode must be 1 (CEF) or 2 (MEF), got {elastic_mode}")
        if nat < 1:
            raise ValueError(f"nat must be >= 1, got {nat}")

        print(f"  Generalized elastic: elastic_mode={elastic_mode} "
              f"({'CEF' if elastic_mode == 1 else 'MEF'}), nat={nat}, nspec={nspec}")

        # Card 6c: lattice parameters
        fvals = reader.read_floats(6)
        latt_a, latt_b, latt_c = fvals[0], fvals[1], fvals[2]
        latt_alpha, latt_beta, latt_gamma = fvals[3], fvals[4], fvals[5]

        print(f"  Lattice: a={latt_a}, b={latt_b}, c={latt_c}, "
              f"alpha={latt_alpha}, beta={latt_beta}, gamma={latt_gamma}")

        # Card 6d: atom types (repeated nat times)
        atom_types = []
        sites = []
        total_atoms_in_cell = 0

        for iat in range(nat):
            # First line: Z A awr_i b_coh sigma_inc npos
            fvals = reader.read_floats(6)
            at_Z = int(fvals[0])
            at_A = int(fvals[1])
            at_awr = fvals[2]
            at_b_coh = fvals[3]       # fm
            at_sigma_inc = fvals[4]   # barns
            at_npos = int(fvals[5])

            # Read fractional coordinates (npos × 3 values)
            coords_flat = reader.read_float_array(at_npos * 3)
            positions = []
            for ip in range(at_npos):
                x = coords_flat[3 * ip]
                y = coords_flat[3 * ip + 1]
                z = coords_flat[3 * ip + 2]
                positions.append((x, y, z))

            sigma_coh = 4.0 * pi * at_b_coh**2 * 0.01  # barns (1 barn = 100 fm²)
            total_atoms_in_cell += at_npos

            atom_types.append({
                'Z': at_Z, 'A': at_A, 'awr': at_awr,
                'b_coh': at_b_coh, 'sigma_inc': at_sigma_inc,
                'sigma_coh': sigma_coh,
                'npos': at_npos, 'positions': positions,
            })

            sites.append(AtomSite(b_coh_fm=at_b_coh, positions=positions))

            print(f"    Atom {iat+1}: Z={at_Z}, A={at_A}, awr={at_awr:.4f}, "
                  f"b_coh={at_b_coh:.4f} fm, sigma_inc={at_sigma_inc:.4f} b, "
                  f"sigma_coh={sigma_coh:.4f} b, npos={at_npos}")

        # Compute fractions
        for at in atom_types:
            at['fraction'] = at['npos'] / total_atoms_in_cell

        # Card 6e: partial phonon spectra (repeated nspec times)
        partial_spectra = []
        for isp in range(nspec):
            # First line: Z A delta_s ni_s
            fvals = reader.read_floats(4)
            sp_Z = int(fvals[0])
            sp_A = int(fvals[1])
            sp_delta = fvals[2]
            sp_ni = int(fvals[3])

            # Read phonon spectrum values
            sp_rho = reader.read_float_array(sp_ni)

            partial_spectra.append({
                'Z': sp_Z, 'A': sp_A,
                'delta': sp_delta, 'ni': sp_ni,
                'rho': sp_rho,
            })

            print(f"    Partial spectrum {isp+1}: Z={sp_Z}, A={sp_A}, "
                  f"delta={sp_delta:.6e}, ni={sp_ni}")

        # Build crystal structure for Bragg edge calculation
        crystal = CrystalStructure(
            a=latt_a, b=latt_b, c=latt_c,
            alpha=latt_alpha, beta=latt_beta, gamma=latt_gamma,
            sites=sites,
        )

        # Match principal scatterer (za) to crystal atom type
        za_Z = int(za) // 1000
        za_A = int(za) % 1000
        principal_atom_idx = None
        for i, at in enumerate(atom_types):
            if at['Z'] == za_Z and at['A'] == za_A:
                principal_atom_idx = i
                break

        if principal_atom_idx is None:
            raise ValueError(
                f"Principal scatterer za={za} (Z={za_Z}, A={za_A}) "
                f"not found in crystal atom types")

        print(f"  Principal scatterer (za={za}) matches atom type "
              f"{principal_atom_idx+1}: Z={za_Z}, A={za_A}, "
              f"fraction={atom_types[principal_atom_idx]['fraction']:.4f}")

        # DC atom selection for CEF (Eq 26 from paper)
        dc_atom_idx = None
        if elastic_mode == 1 and nat > 1:
            min_inc_criterion = float('inf')
            for i, at in enumerate(atom_types):
                f_i = at['fraction']
                criterion = f_i / (1.0 - f_i) * at['sigma_inc']
                if criterion < min_inc_criterion:
                    min_inc_criterion = criterion
                    dc_atom_idx = i

            dc_at = atom_types[dc_atom_idx]
            print(f"  DC atom (min incoherent contribution): type {dc_atom_idx+1}, "
                  f"Z={dc_at['Z']}, A={dc_at['A']}, "
                  f"f_DC={dc_at['fraction']:.4f}")

            if principal_atom_idx == dc_atom_idx:
                print(f"  -> Principal scatterer IS the DC atom -> LTHR=1 (coherent elastic)")
            else:
                print(f"  -> Principal scatterer is NOT the DC atom -> LTHR=2 (incoherent elastic)")

        # Match partial spectra to atom types
        for at in atom_types:
            at['spectrum_idx'] = None
        for i, sp in enumerate(partial_spectra):
            for j, at in enumerate(atom_types):
                if sp['Z'] == at['Z'] and sp['A'] == at['A']:
                    at['spectrum_idx'] = i
                    break

        # Store everything for later use
        crystal_info = {
            'elastic_mode': elastic_mode,
            'nat': nat,
            'nspec': nspec,
            'crystal': crystal,
            'atom_types': atom_types,
            'partial_spectra': partial_spectra,
            'principal_atom_idx': principal_atom_idx,
            'dc_atom_idx': dc_atom_idx,
            'total_atoms_in_cell': total_atoms_in_cell,
        }

    # Card 7: alpha, beta control
    vals = reader.read_ints(3, defaults=[0, 0, 0])
    nalpha, nbeta, lat = vals

    print(f"  nalpha={nalpha}, nbeta={nbeta}, lat={lat}")

    # Card 8: alpha values
    alpha = reader.read_float_array(nalpha)

    # Card 9: beta values
    beta = reader.read_float_array(nbeta)

    # Allocate storage
    ssm = np.zeros((nbeta, nalpha, ntempr))
    ssp = None
    if ncold != 0:
        ssp = np.zeros((nbeta, nalpha, ntempr))

    tempr_arr = np.zeros(ntempr)
    dwpix = np.zeros(ntempr)
    dwp1 = np.zeros(ntempr)
    tempf = np.zeros(ntempr)
    tempf1 = np.zeros(ntempr)

    arat = 1.0

    # Loop over scatterers and temperatures
    isecs = 0
    idone = False
    ssm_principal = None

    while not idone:
        if isecs == 0:
            print("\n  Principal scatterer...")
        else:
            arat = aws / awr
            print(f"\n  Secondary scatterer (alpha scaled by {arat:.3f})...")

        for itemp in range(ntempr):
            temp = reader.read_floats(1)[0]
            tempr_arr[itemp] = abs(temp)
            tev = BK * abs(temp)
            print(f"  Temperature {itemp+1}: {abs(temp):.2f} K")

            if itemp == 0 or temp >= 0.0:
                # Read continuous distribution
                fvals = reader.read_floats(2)
                delta1 = fvals[0]
                ni = int(fvals[1])
                p1 = reader.read_float_array(ni)
                np1 = ni

                fvals = reader.read_floats(3)
                twt = fvals[0]
                c_diff = fvals[1]
                tbeta = fvals[2]

                # Read oscillator data
                nd = reader.read_ints(1)[0]
                bdel = None
                adel = None
                if nd > 0:
                    bdel = reader.read_float_array(nd)
                    adel = reader.read_float_array(nd)

                # Read pair correlation function
                ska = None
                nka = 0
                dka = 0.0
                if nsk > 0 or ncold > 0:
                    fvals = reader.read_floats(2)
                    nka = int(fvals[0])
                    dka = fvals[1]
                    ska = reader.read_float_array(nka)

                # Read coherent fraction
                cfrac = 0.0
                if nsk > 0:
                    cfrac = reader.read_floats(1)[0]

            # Continuous part
            f0, tbar, deltab = contin(ssm[:, :, itemp], alpha, beta,
                                      nalpha, nbeta, lat, arat, tev,
                                      p1, np1, delta1, tbeta, nphon,
                                      iprint, twt, None, None)
            dwpix[itemp] = f0
            tempf[itemp] = tbar * tempr_arr[itemp]

            print(f"    DW lambda = {f0:.6f}, T_eff = {tempf[itemp]:.3f}")

            # Translational part
            if twt > 0.0:
                trans(ssm[:, :, itemp], alpha, beta, nalpha, nbeta,
                      lat, arat, tev, twt, c_diff, tbeta, f0, deltab,
                      tbar, iprint)
                tempf[itemp] = (tbeta * tempf[itemp] + twt * tempr_arr[itemp]) / (tbeta + twt)
                print(f"    After trans: T_eff = {tempf[itemp]:.3f}")

            # Discrete oscillators
            if nd > 0:
                dwpix[itemp], tempf[itemp] = discre(
                    ssm[:, :, itemp], alpha, beta, nalpha, nbeta,
                    lat, arat, tev, twt, tbeta, nd, bdel, adel,
                    dwpix[itemp], tempf[itemp], tempr_arr[itemp], iprint)
                print(f"    After discre: DW = {dwpix[itemp]:.6f}, T_eff = {tempf[itemp]:.3f}")

            # Cold hydrogen/deuterium
            if ncold > 0:
                if ssp is None:
                    ssp = np.zeros((nbeta, nalpha, ntempr))
                coldh(ssm[:, :, itemp], ssp[:, :, itemp],
                      alpha, beta, nalpha, nbeta, lat, arat, tev,
                      twt, tbeta, ncold, ska, nka, dka,
                      tempf[itemp], tempr_arr[itemp], iprint)

            # Skold option
            if nsk == 2 and ncold == 0:
                skold_approx(ssm, alpha, beta, nalpha, nbeta,
                            itemp, ntempr, lat, arat, awr, tev,
                            ska, nka, dka, cfrac)

        # Save or merge for secondary scatterer
        if nss == 0 or b7 > 0.0 or isecs > 0:
            idone = True
        else:
            isecs += 1
            ssm_principal = ssm.copy()
            for itemp in range(ntempr):
                tempf1[itemp] = tempf[itemp]
                dwp1[itemp] = dwpix[itemp]

    # Merge mixed moderator if needed
    if nss != 0 and b7 <= 0.0 and ssm_principal is not None:
        sb = spr * ((1.0 + awr) / awr)**2
        sbs = sps * ((1.0 + aws) / aws)**2
        srat = sbs / sb
        for k in range(ntempr):
            for j in range(nalpha):
                for i in range(nbeta):
                    ssm[i, j, k] = srat * ssm[i, j, k] + ssm_principal[i, j, k]

    # Coherent elastic
    bragg = []
    nedge = 0
    if iel == 10 and crystal_info is not None:
        # Generalized Bragg edge calculation
        # Compute dcutoff from emax: d_min = sqrt(WL2EKIN / (4*emax))

        emax_bragg = 5.0
        dcutoff = sqrt(WL2EKIN / (4.0 * emax_bragg)) * 0.95  # 5% margin
        bragg_data, nedge, species_corr = compute_bragg_edges_general(
            crystal_info['crystal'], emax=emax_bragg, dcutoff=dcutoff)
        # Convert from numpy array to list of (E, delta) tuples
        bragg = [(bragg_data[i, 0], bragg_data[i, 1]) for i in range(nedge)]
        crystal_info['species_corr'] = species_corr
        print(f"  Generalized: found {nedge} Bragg edges below 5 eV")

        # Compute per-species MSD from partial phonon spectra
        _compute_per_species_msd(crystal_info, tempr_arr, ntempr, dwpix)

    elif iel > 0 and iel != 10:
        bragg, nedge = coher(iel, npr, 5.0)
        print(f"  Found {nedge} Bragg edges below 5 eV")

    # Set iel=-1 for incoherent elastic if no translational and no coherent
    if iel == 0 and twt == 0.0:
        iel = -1

    # Determine symmetry
    isym = 0
    if ncold != 0:
        isym = 1
    if isabt == 1:
        isym += 2

    # Read comment cards for MF1/MT451
    comments = reader.read_comment_strings()
    if comments:
        print(f"  Read {len(comments)} comment cards")

    # Write ENDF output
    print(f"\n  Writing ENDF output...")
    write_endf_output(output_file, mat, za, awr, spr, npr, iel, ncold, nss,
                      b7, aws, sps, mss, nalpha, nbeta, lat,
                      alpha, beta, ssm, ssp, tempr_arr, ntempr,
                      dwpix, dwp1, tempf, tempf1, twt, tbeta,
                      bragg, nedge, isym, ilog, smin, iprint,
                      comments=comments, crystal_info=crystal_info)

    print("  LEAPR complete.")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python leapr.py <input_file> <output_file>")
        print("  input_file: NJOY-style LEAPR input")
        print("  output_file: ENDF output file path")
        sys.exit(1)

    run_leapr(sys.argv[1], sys.argv[2])
