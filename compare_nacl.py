#!/usr/bin/env python3
"""Compare Na-in-NaCl vs Cl-in-NaCl TSLs and their weighted sum.

The transport code reconstructs the material scattering as:
  Sigma(E->E') = N_Na * sigma_b_Na * S_Na(a,b) + N_Cl * sigma_b_Cl * S_Cl(a,b)

We compare this weighted sum against a "pseudo NaCl" single-species run
(all atoms treated as principal) to verify the species-aware decomposition.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BK = 8.617333262e-5
WL2EKIN = 0.081804209605330899
OUTDIR = '/Users/ykr/Desktop/Claude_code_playground/NJOY2016'

# NaCl: 4 Na + 4 Cl in conventional cell, equal fractions
F_NA = 0.5   # atom fraction
F_CL = 0.5


def read_endf_mf7mt4(filename):
    from endf_parserpy import EndfParserPy
    parser = EndfParserPy()
    endf_dict = parser.parsefile(filename, include=[(7, 4)])
    mt4 = endf_dict[7][4]
    t0 = float(mt4['T0'])
    awr = float(mt4['AWR'])
    nb = int(mt4['NB'])
    beta = np.array([float(mt4['beta'][i]) for i in range(1, nb + 1)])
    alpha = np.array(mt4['S_table'][1]['alpha'], dtype=float)
    na = len(alpha)
    stored = np.zeros((na, nb))
    for ib in range(nb):
        s_vals = mt4['S_table'][ib + 1]['S']
        for ia in range(na):
            stored[ia, ib] = float(s_vals[ia])
    return alpha, beta, stored, t0, awr


# Load per-species TSLs
a_na, b_na, s_na, t0_na, awr_na = read_endf_mf7mt4(f'{OUTDIR}/nacl_na_test.endf')
a_cl, b_cl, s_cl, t0_cl, awr_cl = read_endf_mf7mt4(f'{OUTDIR}/nacl_cl_test.endf')

kT = BK * t0_na  # same T for both
kT_meV = kT * 1000.0

# sigma_b = spr * ((1+awr)/awr)^2 for each species
spr_na = 3.28   # free XS Na
spr_cl = 11.528  # free XS Cl
sb_na = spr_na * ((1.0 + awr_na) / awr_na)**2
sb_cl = spr_cl * ((1.0 + awr_cl) / awr_cl)**2
print(f"Na: awr={awr_na:.4f}, spr={spr_na}, sigma_b={sb_na:.4f} b")
print(f"Cl: awr={awr_cl:.4f}, spr={spr_cl}, sigma_b={sb_cl:.4f} b")

# Convert stored S(a,b) to S_sym(Q,E) [1/meV] = S_stored * exp(beta/2) / kT_meV
# Note: alpha grids differ between Na and Cl due to different awr
Q_na = np.sqrt(a_na * awr_na * kT / WL2EKIN)  # Q in 1/Ang
Q_cl = np.sqrt(a_cl * awr_cl * kT / WL2EKIN)
E_na = b_na * kT_meV
E_cl = b_cl * kT_meV

Ssym_na = s_na * np.exp(b_na / 2)[np.newaxis, :] / kT_meV
Ssym_cl = s_cl * np.exp(b_cl / 2)[np.newaxis, :] / kT_meV

# Also try loading pseudo-NaCl if available
try:
    a_ps, b_ps, s_ps, t0_ps, awr_ps = read_endf_mf7mt4(
        f'{OUTDIR}/nacl_pseudo_test.endf')
    Q_ps = np.sqrt(a_ps * awr_ps * kT / WL2EKIN)
    E_ps = b_ps * kT_meV
    Ssym_ps = s_ps * np.exp(b_ps / 2)[np.newaxis, :] / kT_meV
    spr_ps = 7.404  # average free XS
    sb_ps = spr_ps * ((1.0 + awr_ps) / awr_ps)**2
    has_pseudo = True
    print(f"Pseudo: awr={awr_ps:.4f}, sigma_b={sb_ps:.4f} b")
except Exception:
    has_pseudo = False
    print("No pseudo-NaCl file found, skipping pseudo comparison")

# ================================================================
# For the weighted sum, we need to evaluate both TSLs at the same Q.
# Since alpha grids differ, interpolate both onto a common Q grid.
# ================================================================
Q_common = np.linspace(0.3, 12.0, 50)

# At each Q, find interpolated S(Q,E) for Na and Cl
# Both share the same beta grid (same structure), so E grids match
assert np.allclose(E_na, E_cl, rtol=1e-4), "Beta grids don't match!"
E = E_na
nb = len(E)


def interp_at_Q(Q_grid, Ssym, Q_target):
    """Interpolate S(Q,E) at a specific Q using nearest-neighbor."""
    ia = np.argmin(np.abs(Q_grid - Q_target))
    return Ssym[ia, :], Q_grid[ia]


# ================================================================
# Plot 1: Individual species at selected Q values
# ================================================================
Q_tests = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
nplots = len(Q_tests)
ncols = 4
nrows = (nplots + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5 * nrows))
for idx, Q_test in enumerate(Q_tests):
    ax = axes.flatten()[idx]
    S_na_q, Q_na_actual = interp_at_Q(Q_na, Ssym_na, Q_test)
    S_cl_q, Q_cl_actual = interp_at_Q(Q_cl, Ssym_cl, Q_test)

    # Transport-code weighted sum: f*sigma_b*S(Q,E) for each species
    # Normalize by total to get effective S per formula unit
    w_na = F_NA * sb_na
    w_cl = F_CL * sb_cl
    w_total = w_na + w_cl
    S_sum = (w_na * S_na_q + w_cl * S_cl_q) / w_total

    ml = (E > 0.3) & (E < 32.0)

    ax.plot(E[ml], S_na_q[ml], 'b-', lw=1.0,
            label=f'Na (Q={Q_na_actual:.2f})')
    ax.plot(E[ml], S_cl_q[ml], 'r-', lw=1.0,
            label=f'Cl (Q={Q_cl_actual:.2f})')
    ax.plot(E[ml], S_sum[ml], 'k-', lw=2.0,
            label='Weighted sum')

    if has_pseudo:
        S_ps_q, Q_ps_actual = interp_at_Q(Q_ps, Ssym_ps, Q_test)
        ax.plot(E_ps[ml], S_ps_q[ml], 'g--', lw=1.5,
                label=f'Pseudo (Q={Q_ps_actual:.2f})')

    ax.set_xlabel('E [meV]')
    ax.set_ylabel('S(Q,E) [1/meV]')
    ax.set_title(f'Q ~ {Q_test:.1f} \u00c5\u207b\u00b9')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Hide unused axes
for idx in range(nplots, nrows * ncols):
    axes.flatten()[idx].set_visible(False)

fig.suptitle('NaCl: Per-species TSLs and weighted sum\n'
             f'(w_Na = {F_NA}*{sb_na:.2f} = {F_NA*sb_na:.2f}, '
             f'w_Cl = {F_CL}*{sb_cl:.2f} = {F_CL*sb_cl:.2f})',
             fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/nacl_comparison.png', dpi=150)
print(f"Saved: {OUTDIR}/nacl_comparison.png")
plt.close(fig)

# ================================================================
# Plot 2: Log scale
# ================================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5 * nrows))
for idx, Q_test in enumerate(Q_tests):
    ax = axes.flatten()[idx]
    S_na_q, Q_na_actual = interp_at_Q(Q_na, Ssym_na, Q_test)
    S_cl_q, Q_cl_actual = interp_at_Q(Q_cl, Ssym_cl, Q_test)

    w_na = F_NA * sb_na
    w_cl = F_CL * sb_cl
    w_total = w_na + w_cl
    S_sum = (w_na * S_na_q + w_cl * S_cl_q) / w_total

    ml = (E > 0.3) & (E < 32.0)

    for arr, col, ls, lw, lab in [
        (S_na_q[ml], 'b', '-', 1.0, f'Na (Q={Q_na_actual:.2f})'),
        (S_cl_q[ml], 'r', '-', 1.0, f'Cl (Q={Q_cl_actual:.2f})'),
        (S_sum[ml], 'k', '-', 2.0, 'Weighted sum'),
    ]:
        pos = arr > 1e-30
        if np.any(pos):
            ax.semilogy(E[ml][pos], arr[pos], color=col, ls=ls, lw=lw,
                        label=lab)

    if has_pseudo:
        S_ps_q, Q_ps_actual = interp_at_Q(Q_ps, Ssym_ps, Q_test)
        pos = S_ps_q[ml] > 1e-30
        if np.any(pos):
            ax.semilogy(E_ps[ml][pos], S_ps_q[ml][pos], 'g--', lw=1.5,
                        label=f'Pseudo (Q={Q_ps_actual:.2f})')

    ax.set_xlabel('E [meV]')
    ax.set_ylabel('S(Q,E) [1/meV]')
    ax.set_title(f'Q ~ {Q_test:.1f} \u00c5\u207b\u00b9 (log)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

for idx in range(nplots, nrows * ncols):
    axes.flatten()[idx].set_visible(False)

fig.suptitle('NaCl: Per-species TSLs and weighted sum (log scale)', fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/nacl_comparison_log.png', dpi=150)
print(f"Saved: {OUTDIR}/nacl_comparison_log.png")
plt.close(fig)

# ================================================================
# Plot 3: Fractional contribution of each species
# ================================================================
fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5 * nrows))
for idx, Q_test in enumerate(Q_tests):
    ax = axes.flatten()[idx]
    S_na_q, _ = interp_at_Q(Q_na, Ssym_na, Q_test)
    S_cl_q, _ = interp_at_Q(Q_cl, Ssym_cl, Q_test)

    w_na = F_NA * sb_na
    w_cl = F_CL * sb_cl
    contrib_na = w_na * S_na_q
    contrib_cl = w_cl * S_cl_q
    total = contrib_na + contrib_cl

    ml = (E > 0.3) & (E < 32.0)
    safe_total = np.where(total > 1e-30, total, 1.0)
    frac_na = np.where(total > 1e-30, contrib_na / safe_total, 0.5)
    frac_cl = np.where(total > 1e-30, contrib_cl / safe_total, 0.5)

    ax.fill_between(E[ml], 0, frac_na[ml], alpha=0.5, color='blue',
                     label='Na fraction')
    ax.fill_between(E[ml], frac_na[ml], 1.0, alpha=0.5, color='red',
                     label='Cl fraction')
    ax.axhline(y=w_na / (w_na + w_cl), color='k', ls='--', lw=0.5,
               label=f'Na weight = {w_na/(w_na+w_cl):.3f}')
    ax.set_xlabel('E [meV]')
    ax.set_ylabel('Fractional contribution')
    ax.set_title(f'Q ~ {Q_test:.1f} \u00c5\u207b\u00b9')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

for idx in range(nplots, nrows * ncols):
    axes.flatten()[idx].set_visible(False)

fig.suptitle('NaCl: Fractional contribution of Na vs Cl to total scattering',
             fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/nacl_fractions.png', dpi=150)
print(f"Saved: {OUTDIR}/nacl_fractions.png")
plt.close(fig)
