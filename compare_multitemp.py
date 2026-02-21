#!/usr/bin/env python3
"""Compare Euphonic vs FLASSH at multiple temperatures for graphite and Be."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = '/Users/ykr/Desktop/Claude_code_playground/NJOY2016'
BK = 8.617333262e-5
WL2EKIN = 0.081804209605330899
THERM = 0.0253


def read_endf_mt4_all_temps(filename):
    """Read MF7/MT4 from an ENDF file and return data for all temperatures.

    Returns
    -------
    alpha, beta : ndarray
    stored_list : list of ndarray (na, nb), one per temperature
    temps : list of float
    awr : float
    lat, lln : int
    """
    from endf_parserpy import EndfParserPy
    parser = EndfParserPy()
    endf_dict = parser.parsefile(filename, include=[(7, 4)])
    mt4 = endf_dict[7][4]
    t0 = float(mt4['T0'])
    awr = float(mt4['AWR'])
    lat = int(mt4['LAT'])
    lln = int(mt4['LLN'])
    nb = int(mt4['NB'])
    beta = np.array([float(mt4['beta'][i]) for i in range(1, nb + 1)])
    alpha = np.array(mt4['S_table'][1]['alpha'], dtype=float)
    na = len(alpha)

    # Build temperature list
    temps = [t0]
    t_dict = mt4.get('T', {})
    for k in sorted(t_dict.keys()):
        temps.append(float(t_dict[k]))

    ntemps = len(temps)
    stored_list = []

    # T0 data from S_table
    stored_t0 = np.zeros((na, nb))
    for ib in range(nb):
        s_vals = mt4['S_table'][ib + 1]['S']
        for ia in range(na):
            stored_t0[ia, ib] = float(s_vals[ia])
    stored_list.append(stored_t0)

    # Additional temperatures from S dict
    if ntemps > 1:
        S_top = mt4['S']
        for it in range(1, ntemps):
            stored_t = np.zeros((na, nb))
            for ia in range(na):
                for ib_key, ib_dict in S_top[ia + 1].items():
                    stored_t[ia, int(ib_key) - 1] = float(ib_dict[it])
            stored_list.append(stored_t)

    return alpha, beta, stored_list, temps, awr, lat, lln


def read_endf_mt2_all_temps(filename):
    """Read MF7/MT2 from an ENDF file for all temperatures.

    Returns
    -------
    E : ndarray of Bragg edge energies
    S_list : list of ndarray, cumulative S at each temperature
    temps : list of float
    lthr : int
    """
    from endf_parserpy import EndfParserPy
    parser = EndfParserPy()
    d = parser.parsefile(filename, include=[(7, 2)])
    mt2 = d[7][2]
    t0 = float(mt2['T0'])
    st = mt2['S_T0_table']
    E = np.array(st['Eint'], dtype=float)
    S_t0 = np.array(st['S'], dtype=float)
    lthr = int(mt2['LTHR'])
    lt = int(mt2.get('LT', 0))

    temps = [t0]
    S_list = [S_t0]

    if lt > 0:
        t_dict = mt2.get('T', {})
        for k in sorted(t_dict.keys()):
            temps.append(float(t_dict[k]))
        S_top = mt2['S']
        n_edges = len(E)
        for it in range(1, len(temps)):
            S_t = np.zeros(n_edges)
            for i in range(n_edges):
                S_t[i] = float(S_top[i + 1][it])
            S_list.append(S_t)

    return E, S_list, temps, lthr


def bragg_xsec(E_edges, S_cumul, E_eval):
    """Compute coherent elastic cross section: sigma(E) = S(E)/E."""
    sigma = np.zeros_like(E_eval)
    for i, e in enumerate(E_eval):
        idx = np.searchsorted(E_edges, e, side='right') - 1
        if idx >= 0:
            sigma[i] = S_cumul[idx] / e
    return sigma


def match_temps(temps_euph, temps_flassh):
    """Find matching temperature pairs between Euphonic and FLASSH."""
    pairs = []
    for ie, te in enumerate(temps_euph):
        best_if = np.argmin([abs(tf - te) for tf in temps_flassh])
        if abs(temps_flassh[best_if] - te) < 5.0:  # within 5K
            pairs.append((ie, best_if, te, temps_flassh[best_if]))
    return pairs


# ================================================================
# MT4 inelastic comparison at multiple temperatures
# ================================================================
def plot_mt4_multitemp(alpha_e, beta_e, stored_e_list, temps_e, awr_e,
                       alpha_f, beta_f, stored_f_list, temps_f, awr_f,
                       material_name, prefix,
                       lat_e=1, lat_f=1, lln_e=0, lln_f=0):
    """Plot MT4 S(Q,E) comparison at multiple temperatures."""

    pairs = match_temps(temps_e, temps_f)
    ntemps = len(pairs)
    print(f"\n{material_name}: {ntemps} matching temperature pairs "
          f"(LAT: euph={lat_e}, flassh={lat_f})")
    for ie, iff, te, tf in pairs:
        print(f"  Euphonic T={te}K <-> FLASSH T={tf}K")

    # Select Q values for comparison
    Q_tests = [1.0, 3.0, 7.0, 14.0]

    # ================================================================
    # Figure: S(Q,E) at selected Q for all temperatures
    # ================================================================
    for Q_test in Q_tests:
        ncols = min(4, ntemps)
        nrows = (ntemps + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        for idx, (ie, iff, te, tf) in enumerate(pairs):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            stored_e = stored_e_list[ie]
            stored_f = stored_f_list[iff]

            kT_eV = BK * te
            kT_meV = kT_eV * 1000.0

            # LAT=1: beta in kT_thermal units, sc = THERM/kT
            # LAT=0 or other: beta in kT units, sc = 1
            sc_e = THERM / kT_eV if lat_e == 1 else 1.0
            sc_f = THERM / kT_eV if lat_f == 1 else 1.0

            # Q and E from Euphonic grid (used for axis labels)
            # For LAT=1: Q = sqrt(alpha * awr * THERM / WL2EKIN) (T-independent)
            # For LAT=0: Q = sqrt(alpha * awr * kT / WL2EKIN) (T-dependent)
            kT_alpha_e = THERM if lat_e == 1 else kT_eV
            kT_alpha_f = THERM if lat_f == 1 else kT_eV
            kT_beta_e = THERM * 1000.0 if lat_e == 1 else kT_meV
            kT_beta_f = THERM * 1000.0 if lat_f == 1 else kT_meV

            Q_e = np.sqrt(alpha_e * awr_e * kT_alpha_e / WL2EKIN)
            E_e = beta_e * kT_beta_e
            Q_f = np.sqrt(alpha_f * awr_f * kT_alpha_f / WL2EKIN)
            E_f = beta_f * kT_beta_f

            ia_e = np.argmin(np.abs(Q_e - Q_test))
            ia_f = np.argmin(np.abs(Q_f - Q_test))
            ml_e = (E_e > 0.5) & (E_e < 200.0)
            ml_f = (E_f > 0.5) & (E_f < 200.0)

            S_e = stored_e[ia_e, :] * np.exp(beta_e * sc_e / 2) / kT_beta_e
            S_f = stored_f[ia_f, :] * np.exp(beta_f * sc_f / 2) / kT_beta_f

            ax.plot(E_e[ml_e], S_e[ml_e], 'b-', lw=1.2, label='Euphonic')
            ax.plot(E_f[ml_f], S_f[ml_f], 'r--', lw=0.8, label='FLASSH')

            area_e = np.trapezoid(S_e[ml_e], E_e[ml_e]) if np.any(ml_e) else 0
            area_f = np.trapezoid(S_f[ml_f], E_f[ml_f]) if np.any(ml_f) else 0
            ratio = area_e / area_f if area_f > 0 else 0

            ax.set_xlabel('E [meV]')
            ax.set_ylabel('S(Q,E) [1/meV]')
            ax.set_title(f'T={te:.0f}K  (ratio={ratio:.3f})')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(ntemps, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        fig.suptitle(f'{material_name} S(Q,E) at Q\u2248{Q_test} \u00c5\u207b\u00b9: '
                     f'Euphonic vs FLASSH', fontsize=14)
        fig.tight_layout()
        fname = f'{OUTDIR}/{prefix}_mt4_Q{Q_test:.0f}_multitemp.png'
        fig.savefig(fname, dpi=150)
        print(f"Saved: {fname}")
        plt.close(fig)

    # ================================================================
    # Figure: integrated ratio vs Q at all temperatures
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.plasma(np.linspace(0, 0.9, ntemps))

    for idx, (ie, iff, te, tf) in enumerate(pairs):
        stored_e = stored_e_list[ie]
        stored_f = stored_f_list[iff]

        kT_eV = BK * te
        kT_meV = kT_eV * 1000.0

        sc_e = THERM / kT_eV if lat_e == 1 else 1.0
        sc_f = THERM / kT_eV if lat_f == 1 else 1.0
        kT_alpha_e = THERM if lat_e == 1 else kT_eV
        kT_alpha_f = THERM if lat_f == 1 else kT_eV
        kT_beta_e = THERM * 1000.0 if lat_e == 1 else kT_meV
        kT_beta_f = THERM * 1000.0 if lat_f == 1 else kT_meV

        Q_e = np.sqrt(alpha_e * awr_e * kT_alpha_e / WL2EKIN)
        Q_f = np.sqrt(alpha_f * awr_f * kT_alpha_f / WL2EKIN)
        E_e = beta_e * kT_beta_e
        E_f = beta_f * kT_beta_f
        ml_e = (E_e > 0.5) & (E_e < 100.0)
        ml_f = (E_f > 0.5) & (E_f < 100.0)

        na = len(alpha_e)
        ratios = np.zeros(na)
        for ia in range(na):
            ia_f = np.argmin(np.abs(Q_f - Q_e[ia]))
            S_ev = stored_e[ia, :] * np.exp(beta_e * sc_e / 2) / kT_beta_e
            S_fv = stored_f[ia_f, :] * np.exp(beta_f * sc_f / 2) / kT_beta_f
            area_e_v = np.trapezoid(S_ev[ml_e], E_e[ml_e])
            area_f_v = np.trapezoid(S_fv[ml_f], E_f[ml_f])
            ratios[ia] = area_e_v / area_f_v if area_f_v > 1e-30 else 0

        ax.plot(Q_e, ratios, '-', lw=1.0, color=colors[idx], label=f'T={te:.0f}K')

    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Q [\u00c5\u207b\u00b9]')
    ax.set_ylabel('Integrated ratio Euphonic/FLASSH')
    ax.set_title(f'{material_name}: Euphonic/FLASSH Ratio vs Q at Multiple Temperatures')
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f'{OUTDIR}/{prefix}_mt4_ratio_multitemp.png'
    fig.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close(fig)

    # ================================================================
    # Figure: stored S_s(alpha,beta) at one alpha across all temperatures
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    alpha_targets = [0.5, 2.0, 10.0, 50.0]

    for pidx, alpha_target in enumerate(alpha_targets):
        ax = axes[pidx]
        ia = np.argmin(np.abs(alpha_e - alpha_target))

        for idx, (ie, iff, te, tf) in enumerate(pairs):
            stored_e = stored_e_list[ie]
            stored_f = stored_f_list[iff]
            mb = beta_e > 0.01

            e_vals = stored_e[ia, mb]
            f_vals = stored_f[ia, mb]
            e_pos = e_vals > 1e-30
            f_pos = f_vals > 1e-30

            color = colors[idx]
            if np.any(e_pos):
                ax.semilogy(beta_e[mb][e_pos], e_vals[e_pos], '-',
                           lw=1.0, color=color, label=f'E T={te:.0f}K' if pidx == 0 else '')
            if np.any(f_pos):
                ax.semilogy(beta_f[mb][f_pos], f_vals[f_pos], '--',
                           lw=0.6, color=color, label=f'F T={tf:.0f}K' if pidx == 0 else '')

        ax.set_xlabel('\u03b2')
        ax.set_ylabel('Stored S_s(\u03b1,\u03b2)')
        ax.set_title(f'\u03b1 = {alpha_e[ia]:.4f}')
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f'{material_name} Stored S_s(\u03b1,\u03b2) at Multiple Temperatures\n'
                 f'(solid=Euphonic, dashed=FLASSH)', fontsize=13)
    fig.tight_layout()
    fname = f'{OUTDIR}/{prefix}_mt4_stored_multitemp.png'
    fig.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close(fig)


# ================================================================
# MT2 coherent elastic comparison at multiple temperatures
# ================================================================
def plot_mt2_multitemp(E_e, S_e_list, temps_e, E_f, S_f_list, temps_f,
                       material_name, prefix):
    """Plot MT2 coherent elastic comparison at multiple temperatures."""

    pairs = match_temps(temps_e, temps_f)
    ntemps = len(pairs)
    print(f"\n{material_name} MT2: {ntemps} matching temperature pairs")

    E_eval = np.geomspace(1e-4, 5.0, 5000)

    # ================================================================
    # Figure: cross section at all temperatures
    # ================================================================
    ncols = min(4, ntemps)
    nrows = (ntemps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, (ie, iff, te, tf) in enumerate(pairs):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        sigma_e = bragg_xsec(E_e, S_e_list[ie], E_eval)
        sigma_f = bragg_xsec(E_f, S_f_list[iff], E_eval)

        ax.loglog(E_eval * 1000, sigma_e, 'b-', lw=0.8, label='Euphonic')
        ax.loglog(E_eval * 1000, sigma_f, 'r--', lw=0.8, label='FLASSH')
        ax.set_xlabel('E [meV]')
        ax.set_ylabel('\u03c3 [barn]')
        ax.set_title(f'T={te:.0f}K')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(ntemps, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f'{material_name} MT2 Coherent Elastic: Euphonic vs FLASSH', fontsize=14)
    fig.tight_layout()
    fname = f'{OUTDIR}/{prefix}_mt2_multitemp.png'
    fig.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close(fig)

    # ================================================================
    # Figure: cumulative S at all temperatures (overlaid)
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.plasma(np.linspace(0, 0.9, ntemps))

    for idx, (ie, iff, te, tf) in enumerate(pairs):
        color = colors[idx]
        ax.step(E_e * 1000, S_e_list[ie], '-', lw=0.8, color=color,
                where='post', label=f'E T={te:.0f}K')
        ax.step(E_f * 1000, S_f_list[iff], '--', lw=0.6, color=color,
                where='post', label=f'F T={tf:.0f}K')

    ax.set_xlabel('E [meV]')
    ax.set_ylabel('Cumulative S [eV\u00b7barn]')
    ax.set_title(f'{material_name} Cumulative Bragg Structure Factor at Multiple Temperatures\n'
                 f'(solid=Euphonic, dashed=FLASSH)')
    ax.set_xlim(0, 500)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f'{OUTDIR}/{prefix}_mt2_cumS_multitemp.png'
    fig.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close(fig)


# ================================================================
# Main
# ================================================================
print("=" * 60)
print("GRAPHITE: Multi-temperature comparison")
print("=" * 60)

print("Loading Euphonic output...")
alpha_ge, beta_ge, stored_ge, temps_ge, awr_ge, lat_ge, lln_ge = \
    read_endf_mt4_all_temps(f'{OUTDIR}/graphite_test.endf')
print(f"  Temperatures: {temps_ge}")

print("Loading FLASSH reference...")
alpha_gf, beta_gf, stored_gf, temps_gf, awr_gf, lat_gf, lln_gf = \
    read_endf_mt4_all_temps(f'{OUTDIR}/tsl-graphiteSd_reference.endf')
print(f"  Temperatures: {temps_gf}")

plot_mt4_multitemp(alpha_ge, beta_ge, stored_ge, temps_ge, awr_ge,
                   alpha_gf, beta_gf, stored_gf, temps_gf, awr_gf,
                   'Graphite', 'graphite',
                   lat_e=lat_ge, lat_f=lat_gf, lln_e=lln_ge, lln_f=lln_gf)

print("\nLoading MT2 data...")
E_ge2, S_ge2, temps_ge2, lthr_ge2 = read_endf_mt2_all_temps(f'{OUTDIR}/graphite_test.endf')
E_gf2, S_gf2, temps_gf2, lthr_gf2 = read_endf_mt2_all_temps(
    f'{OUTDIR}/tsl-graphiteSd_reference.endf')
print(f"  Euphonic MT2 temps: {temps_ge2}")
print(f"  FLASSH  MT2 temps: {temps_gf2}")

plot_mt2_multitemp(E_ge2, S_ge2, temps_ge2, E_gf2, S_gf2, temps_gf2,
                   'Graphite', 'graphite')


print("\n" + "=" * 60)
print("BERYLLIUM: Multi-temperature comparison")
print("=" * 60)

print("Loading Euphonic output...")
alpha_be, beta_be, stored_be, temps_be, awr_be, lat_be, lln_be = \
    read_endf_mt4_all_temps(f'{OUTDIR}/be_test.endf')
print(f"  Temperatures: {temps_be}")

print("Loading FLASSH reference...")
alpha_bf, beta_bf, stored_bf, temps_bf, awr_bf, lat_bf, lln_bf = \
    read_endf_mt4_all_temps(f'{OUTDIR}/tsl-Be-metal+Sd.endf')
print(f"  Temperatures: {temps_bf}")

plot_mt4_multitemp(alpha_be, beta_be, stored_be, temps_be, awr_be,
                   alpha_bf, beta_bf, stored_bf, temps_bf, awr_bf,
                   'Be metal', 'be',
                   lat_e=lat_be, lat_f=lat_bf, lln_e=lln_be, lln_f=lln_bf)

print("\nLoading MT2 data...")
E_be2, S_be2, temps_be2, lthr_be2 = read_endf_mt2_all_temps(f'{OUTDIR}/be_test.endf')
E_bf2, S_bf2, temps_bf2, lthr_bf2 = read_endf_mt2_all_temps(
    f'{OUTDIR}/tsl-Be-metal+Sd.endf')
print(f"  Euphonic MT2 temps: {temps_be2}")
print(f"  FLASSH  MT2 temps: {temps_bf2}")

plot_mt2_multitemp(E_be2, S_be2, temps_be2, E_bf2, S_bf2, temps_bf2,
                   'Be metal', 'be')
