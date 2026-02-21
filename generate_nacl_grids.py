#!/usr/bin/env python3
"""Generate comprehensive alpha/beta grids for NaCl LEAPR inputs.

Uses the approach from Generate_generic_NJOY_input_from_NCMAT_file.py:
- Beta grid: logarithmic lower tail, linear in phonon spectrum region, logarithmic upper tail
- Alpha grid: alpha = 4*beta/A (recoil relation)
"""
import numpy as np
import sys
import os

sys.path.insert(0, '/Users/ykr/Desktop/Claude_code_playground/Euphonic')

from euphonic import ForceConstants, Quantity
from euphonic.util import mp_grid

# Parameters
phonopy_path = '/Users/ykr/Desktop/Claude_code_playground/Euphonic/tests_and_analysis/test/data/phonopy_files/NaCl'
number_of_phonon_spectra_points_delta_e = 300
lower_beta_grid_points = 50
upper_beta_grid_points = 20
temperature = 300.0  # K
kT_eV = 8.617333262e-5 * temperature
OUTDIR = '/Users/ykr/Desktop/Claude_code_playground/NJOY2016'

# Load force constants and compute phonon frequencies
fc = ForceConstants.from_phonopy(path=phonopy_path)
mesh = [10, 10, 10]
qpts = mp_grid(mesh)
modes = fc.calculate_qpoint_phonon_modes(qpts, asr='reciprocal')
freqs_ev = modes.frequencies.to('eV').magnitude

freq_max = np.max(freqs_ev[freqs_ev > 0])
print(f"Max phonon frequency: {freq_max*1000:.2f} meV ({freq_max:.6f} eV)")

# Create phonon energy grid
delta_e = freq_max / number_of_phonon_spectra_points_delta_e
erg_phonon = np.arange(1, number_of_phonon_spectra_points_delta_e) * delta_e

# Convert to beta = E / kT
beta_linear = erg_phonon / kT_eV

# Logarithmic tails
beta_lower = np.geomspace(1e-9 / kT_eV, beta_linear[0], lower_beta_grid_points)
beta_upper_max = 5.0 / kT_eV
beta_upper = np.geomspace(beta_linear[-1], beta_upper_max, upper_beta_grid_points)

# Combine
beta_without_zero = np.concatenate((beta_lower[:-2], beta_linear, beta_upper[1:]))
beta = np.concatenate(([0.0], beta_without_zero))

print(f"Beta grid: {len(beta)} points, range [{beta[1]:.6e}, {beta[-1]:.4f}]")

# Alpha grids
mass_na = 22.9898
mass_cl = 34.9689
alpha_na = 4.0 * beta_without_zero / mass_na
alpha_cl = 4.0 * beta_without_zero / mass_cl

print(f"Na: nalpha={len(alpha_na)}, nbeta={len(beta)}")
print(f"Cl: nalpha={len(alpha_cl)}, nbeta={len(beta)}")


def format_array(arr, values_per_line=6):
    """Format array for LEAPR input."""
    lines = []
    for i in range(0, len(arr), values_per_line):
        chunk = arr[i:i+values_per_line]
        lines.append(' '.join(f'{v:.6e}' for v in chunk))
    return '\n'.join(lines) + ' /'


def write_leapr(filename, title, za, awr, spr, alpha, beta,
                atom_lines, ndir=100, mesh='10 10 10', nbin=200):
    """Write a complete .leapr file."""
    nalpha = len(alpha)
    nbeta = len(beta)
    phonopy_yaml = f'{phonopy_path}/phonopy.yaml'

    with open(filename, 'w') as f:
        f.write('20 /\n')
        f.write(f"'{title}' /\n")
        f.write('1 1 100 /\n')
        f.write(f'100 {za} 0 0 /\n')
        f.write(f'{awr} {spr} 0 10 0 0 /\n')
        f.write('0 0 0 0 0 /\n')
        # atom_lines includes elastic_mode line + all atom definitions
        f.write(atom_lines)
        f.write(f'{ndir} {mesh} {nbin} 0 0 /\n')
        f.write(f"'{phonopy_yaml}' /\n")
        f.write(f'{nalpha} {nbeta} 0 /\n')
        f.write(format_array(alpha) + '\n')
        f.write(format_array(beta) + '\n')
        f.write('300.0 /\n')
        f.write('0.0 0.0 1.0 /\n')
        f.write('0 /\n')


# Na in NaCl (2 atom types, Na is principal)
na_atoms = (
    '2 2 0 1 /\n'
    '5.6903 5.6903 5.6903 90.0 90.0 90.0 /\n'
    '11 23 22.9898 3.63 1.62 4 /\n'
    '0.0 0.0 0.0  0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0 /\n'
    '17 35 34.9689 11.65 4.7 4 /\n'
    '0.5 0.5 0.5  0.5 0.0 0.0  0.0 0.5 0.0  0.0 0.0 0.5 /\n'
)

# Cl in NaCl (2 atom types, Cl is principal)
cl_atoms = (
    '2 2 0 1 /\n'
    '5.6903 5.6903 5.6903 90.0 90.0 90.0 /\n'
    '11 23 22.9898 3.63 1.62 4 /\n'
    '0.0 0.0 0.0  0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0 /\n'
    '17 35 34.9689 11.65 4.7 4 /\n'
    '0.5 0.5 0.5  0.5 0.0 0.0  0.0 0.5 0.0  0.0 0.0 0.5 /\n'
)

# Pseudo NaCl (1 atom type, all 8 atoms as Na)
pseudo_atoms = (
    '2 1 0 1 /\n'
    '5.6903 5.6903 5.6903 90.0 90.0 90.0 /\n'
    '11 23 22.9898 3.63 1.62 8 /\n'
    '0.0 0.0 0.0  0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0  '
    '0.5 0.5 0.5  0.5 0.0 0.0  0.0 0.5 0.0  0.0 0.0 0.5 /\n'
)

write_leapr(f'{OUTDIR}/nacl_na_test.leapr',
            'Na in NaCl - ncoh_inel=1 comprehensive grid',
            11023, 22.9898, 3.28, alpha_na, beta, na_atoms)

write_leapr(f'{OUTDIR}/nacl_cl_test.leapr',
            'Cl in NaCl - ncoh_inel=1 comprehensive grid',
            17035, 34.9689, 11.528, alpha_cl, beta, cl_atoms)

write_leapr(f'{OUTDIR}/nacl_pseudo_test.leapr',
            'Pseudo-NaCl single species comprehensive grid',
            11023, 22.9898, 3.28, alpha_na, beta, pseudo_atoms)

print(f"\nWrote: nacl_na_test.leapr ({len(alpha_na)} x {len(beta)})")
print(f"Wrote: nacl_cl_test.leapr ({len(alpha_cl)} x {len(beta)})")
print(f"Wrote: nacl_pseudo_test.leapr ({len(alpha_na)} x {len(beta)})")
