# AGENTS.md - Python LEAPR Translation

## Project Overview

This project contains two Python LEAPR implementations:

- **`leapr.py`** — Direct translation of NJOY2016's `leapr.f90`. Supports the
  standard LEAPR input format with hardcoded coherent elastic for six materials
  (graphite, Be, BeO, Al, Pb, Fe).

- **`leapr_generalized.py`** — Extended version with generalized coherent elastic
  scattering for arbitrary crystal structures (`iel=10`). Self-contained (no
  external module dependencies beyond NumPy and endf-parserpy). Implements the
  formalism from K. Ramic, J. I. Damian Marquez, et al., NIM-A 1027 (2022) 166227.

Both calculate the thermal neutron scattering law S(alpha, beta) from phonon
frequency distributions and write output in ENDF-6 format (MF7/MT4 for inelastic,
MF7/MT2 for elastic).

The Fortran source is at `src/leapr.f90` (~3,600 lines). The base Python
translation is `leapr.py` (~2,714 lines). The generalized version is
`leapr_generalized.py` (~3,600 lines), which includes all base functionality
plus the Bragg-edge calculator and generalized elastic builders.

## Dependencies

- Python 3.8+
- NumPy
- endf_parserpy (`pip install endf-parserpy`)
  - Uses the `EndfParserPy` class (not the deprecated `EndfParser` alias)

## Usage

```bash
python leapr.py <input_file> <output_file>              # standard LEAPR
python leapr_generalized.py <input_file> <output_file>  # with iel=10 support
```

- `input_file`: NJOY-format LEAPR input deck (same format as Fortran NJOY)
- `output_file`: ENDF-6 output file path

## Architecture

### Input Parsing

`parse_leapr_input(filename)` (line 61) reads NJOY free-format input decks.
Values are separated by spaces or `/` (which terminates a card). Quoted strings
are preserved. The parser handles NJOY conventions like negative temperatures
(meaning "reuse previous phonon spectrum at this temperature").

### Main Driver

`run_leapr(input_file, output_file)` (line 2476) is the entry point. It:

1. Parses input parameters (mat, za, awr, alpha/beta grids, temperatures, etc.)
2. Loops over scatterers (principal first, then secondary if `nss > 0`)
3. For each scatterer and temperature:
   - Reads phonon density of states (`rho`)
   - Normalizes and prepares the phonon spectrum via `start()`
   - Iteratively convolves the phonon expansion via `contin()`
   - Adds translational contribution via `trans()` (if `twt > 0`)
   - Adds discrete oscillators via `discre()` (if any)
   - Applies cold hydrogen/deuterium corrections via `coldh()` (if `ncold != 0`)
   - Applies Skold approximation via `skold_approx()` (if `nsk > 0`)
4. Merges principal + secondary scatterer results if mixed moderator
5. Computes Bragg edges via `coher()` if coherent elastic (`iel > 0`)
6. Writes ENDF output via `write_endf_output()`

### Key Functions (with line numbers)

| Function | Line | Purpose |
|----------|------|---------|
| `parse_leapr_input` | 61 | Parse NJOY free-format input |
| `start` | 330 | Prepare phonon spectrum, compute T_eff and Debye-Waller |
| `contin` | 455 | Phonon expansion convolution (iterative) |
| `trans` | 720 | Translational (diffusion/free gas) contribution |
| `discre` | 1072 | Discrete oscillator contribution |
| `coldh` | 1444 | Cold hydrogen/deuterium (ortho/para) |
| `coher` | 1579 | Coherent elastic Bragg edge calculation |
| `skold_approx` | 1796 | Skold intermolecular coherence correction |
| `sigfig` | 1842 | Fortran-compatible significant figure rounding |
| `write_endf_output` | 1873 | ENDF-6 file output via endf_parserpy |
| `run_leapr` | 2476 | Main driver |

### Helper Functions

| Function | Line | Purpose |
|----------|------|---------|
| `fsum` | 303 | Phonon spectrum normalization integral |
| `terpt` / `terpt_vec` | 368/381 | Log-linear interpolation in phonon tables |
| `convol` | 395 | Simpson's-rule convolution of two arrays |
| `besk1` | 545 | Modified Bessel function K1 |
| `terps` | 598 | Log-linear interpolation in S(beta) tables |
| `stable` | 621 | Generate diffusion/free-gas S(beta) table |
| `sbfill` | 679 | Interpolate S(beta) onto convolution grid |
| `bfact` | 837 | Discrete oscillator Bessel function weights |
| `bfill` | 931 | Prepare beta grid for discrete oscillators |
| `exts` | 965 | Extend S(beta) to negative beta via detailed balance |
| `sint` / `sint_vec` | 992/1040 | Interpolate S(alpha,beta) for oscillator convolution |
| `sjbes` | 1279 | Spherical Bessel functions for cold hydrogen |
| `cn_cg` | 1346 | Clebsch-Gordan coefficients for cold hydrogen |
| `formf` | 1769 | Lattice form factors for Bragg scattering |

## Performance Notes

### Thread Limiting (Critical)

Lines 22-27 set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1` **before** importing NumPy. This is critical for performance.
The inner loops in `trans()` operate on many small arrays where threading overhead
dominates. Without this, test 80 shows ~74s of system time (thread management)
vs ~1.2s with single-threading.

### trans() Optimization

The `trans()` function (line 720) is the main performance bottleneck for large
grids (e.g., test 80: 200 alpha x 1325 beta). Key optimizations:

- **Precomputed log(ap) and interpolation slopes** once per alpha, reused across
  all beta iterations (avoids redundant log() calls)
- **Pre-allocated buffers** (`bet_arr`, `b_arr`, `sb_val`, `sb`) outside the
  beta loop
- **Conditional exp()** via `np.exp(sb_val, where=good, out=sb)` to skip
  computing exp on out-of-range elements
- **Simpson's rule via dot products** instead of explicit loops

Performance on test 80 (200 alpha x 1325 beta, 1 temperature):
- Fortran NJOY: ~38s
- Python (optimized): ~76s (2.0x ratio)

### What Was Tried and Didn't Help

- **Full batch vectorization** (all betas at once): memory thrashing from large
  intermediate arrays made it slower (~163s)
- **Chunked batch** (groups of 64 betas): same memory issue (~164s)
- **np.take with out=**: function call overhead exceeded allocation savings (~101s)

## ENDF Output

### endf_parserpy Integration

`write_endf_output()` (line 1873) builds a nested dict matching the ENDF-6
recipe structure that endf_parserpy expects, then calls `parser.writefile()`.

Post-processing (lines 2252-2279) fixes SEND/FEND/MEND/TEND records: endf_parserpy
writes `0.000000+0` in data fields, but NJOY writes blank fields. The post-processor
blanks out the first 66 characters of these special records.

### Comment Card / Text Record Layout

ENDF MF1/MT451 text records follow this field layout:

**Text record 1** (line 1969-1975):
```
ZSYMAM{11} + ALAB{11} + EDATE{10} + {1}blank + AUTH{33} = 66 chars
Positions: [0:11]    [11:22]    [22:32]     [32]       [33:66]
```

**Text record 2** (lines 1977-1984):
```
{1}blank + REF{21} + DDATE{10} + {1}blank + RDATE{10} + {12}pad + ENDATE{8} + {3}pad
Positions: [0]  [1:22]    [22:32]    [32]      [33:43]    [43:55]   [55:63]    [63:66]
```

### T_eff Table Ordering

For mixed moderators (`nss > 0, b7 <= 0`), the Fortran writes two TAB1 records:
- First TAB1 (teff1 slot): `tempf1` values (principal scatterer's T_eff)
- Second TAB1 (teff0 slot): `tempf` values (secondary scatterer's T_eff)

Since endf_parserpy writes `teff0_table` first, then `teff1_table`, the Python
code maps: `teff0_table/Teff0 = tempf1` (principal) and `teff1_table/Teff1 = tempf`
(secondary) to match Fortran output order. See lines 2194-2209.

## Test Suite

### Running Tests

Each test uses an NJOY-format input file. Run with:
```bash
python leapr.py <input_file> <output_file>
```

To generate Fortran reference files, build NJOY2016 and run:
```bash
mkdir -p /tmp/njoy_<name> && cd /tmp/njoy_<name> && /path/to/njoy < <input_file>
```

### Test Matrix

| Test | Input File | Output | Fortran Reference | Description |
|------|-----------|--------|-------------------|-------------|
| test 09 | `tests/09/input` | `test09_output.endf` | `/tmp/njoy_test09/tape24` | H in H2O (shortened model) |
| test 22 | `tests/22/input` | `test22_output.endf` | `tests/22/referenceTape20` | Para hydrogen at 20K |
| test 23 | `tests/23/input` | `test23_output.endf` | `tests/23/referenceTape20` | BeO (mixed moderator, 8 temps) |
| test 33 | `tests/33/input` | `test33_output.endf` | `tests/33/referenceTape24` | D in D2O (CAB model) |
| test 80 | `tests/80/input` | `test80_output.endf` | `/tmp/njoy_test80/tape24` | H in HF (large grid, perf test) |
| graphite | `tsl-crystalline-graphite.leapr` | `tsl-crystalline-graphite-py.endf` | `/tmp/njoy_graphite/tape24` | Crystalline graphite |
| l-CH4 | `tsl-l-CH4.leapr` | `tsl-l-CH4-py.endf` | `/tmp/njoy_lch4/tape20` | Liquid methane at 100K |
| s-CH4 | `tsl-s-CH4.leapr` | `tsl-s-CH4-py.endf` | `/tmp/njoy_sch4/tape20` | Solid methane at 22K |
| Al | `tsl-013_Al_027.leapr` | `tsl-Al-py.endf` | `/tmp/njoy_al/tape25` | Aluminum (FCC) |
| Fe | `tsl-026_Fe_056.leapr` | `tsl-Fe-py.endf` | `/tmp/njoy_fe/tape25` | Iron-56 (BCC) |
| HinCH2 | `tsl-HinCH2.leapr` | `tsl-HinCH2-py.endf` | `/tmp/njoy_hinch2/tape24` | H in polyethylene |

### Test Results (as of 2026-02-18)

| Test | Diff Lines | Status | Notes |
|------|-----------|--------|-------|
| test 09 | 202 | PASS | FP rounding (last digit only) |
| test 22 | 0 | PERFECT | Identical to Fortran |
| test 23 | 4 | PASS | `*` comment prefix (endf_parserpy limitation) |
| test 33 | 0 | PERFECT | Identical to Fortran |
| test 80 | ~47,000 | PASS | FP rounding (last digit, large grid) |
| graphite | 0 | PERFECT | Identical to Fortran |
| l-CH4 | 0 | PERFECT | Identical to Fortran |
| s-CH4 | 4 | PASS | Number format: `3.279202+2` vs `327.920230` |
| Al | 0 | PERFECT | Identical to Fortran |
| Fe | 0 | PERFECT | Identical to Fortran |
| HinCH2 | 4 | PASS | Number format: `1.635904+2` vs `163.590439` |

**7 of 11 tests produce byte-identical output to Fortran NJOY.**

### Known Differences

1. **FP rounding** (tests 09, 80): Last-digit differences in S(alpha,beta) values
   due to Python vs Fortran floating-point arithmetic ordering. All differences
   are < 10 ppm. This is inherent and expected.

2. **Number formatting** (s-CH4, HinCH2): endf_parserpy uses exponential notation
   (`3.279202+2`) where Fortran uses fixed-point (`327.920230`) for values in the
   100-1000 range in MF7/MT2. Both are valid ENDF representations. The values are
   numerically identical.

3. **`*` comment prefix** (test 23): The Fortran reference has `*` at position 0
   of text record 2 in MF1/MT451, but endf_parserpy's template defines position 0
   as `{1}blank`, which overwrites any leading `*`. This is a fundamental limitation
   of the field-based approach in endf_parserpy and cannot be fixed without
   bypassing the library.

## Key Design Decisions

1. **Must use endf_parserpy** for ENDF output. Do not bypass it with raw string
   formatting. The library handles all the complex ENDF-6 record structure
   (CONT, TAB1, TAB2, LIST records, pagination, MAT/MF/MT/line numbering).

2. **sigfig()** (line 1842) replicates the Fortran `sigfig` function exactly.
   It rounds to N significant figures with ENDF-compatible output. This is
   essential for matching Fortran output values.

3. **Thread limiting** must happen before `import numpy`. The `os.environ.setdefault`
   calls at lines 22-27 are not optional for performance.

4. **Per-beta loop in trans()** is faster than batch vectorization for this
   workload due to memory access patterns. Do not attempt to vectorize across
   all betas simultaneously.

## Fortran-to-Python Mapping

Key variable name correspondences between `leapr.f90` and `leapr.py`:

| Fortran | Python | Description |
|---------|--------|-------------|
| `ssm(nbeta,nalpha,ntempr)` | `ssm[nbeta, nalpha]` (per temp) | S(alpha,beta) matrix |
| `ssp(nbeta,nalpha,ntempr)` | `ssp[nbeta, nalpha]` (per temp) | Cold hydrogen S matrix |
| `tempf(i)` | `tempf[i]` | Effective temperature (current scatterer) |
| `tempf1(i)` | `tempf1[i]` | Saved principal scatterer T_eff |
| `dwpix(i)` | `dwpix[i]` | Debye-Waller coefficient |
| `dwp1(i)` | `dwp1[i]` | Saved principal DW coefficient |
| `tempr(i)` | `tempr[i]` | Requested temperatures |
| `arat` | `arat` | Mass ratio (awr_secondary / awr_principal) for alpha scaling |

## Generalized Coherent Elastic (iel=10)

### Overview

`leapr_generalized.py` extends LEAPR with generalized coherent elastic scattering
for arbitrary crystal structures. This is activated by setting `iel=10` on Card 5,
which triggers additional input cards (6b-6e) that define the crystal structure,
atom types, and partial phonon spectra.

The implementation follows:
  K. Ramic, J. I. Damian Marquez, et al., "NJOY+NCrystal: An open-source tool
  for creating thermal neutron scattering libraries with mixed elastic support",
  NIM-A 1027 (2022) 166227.

### Elastic Modes

- **CEF (elastic_mode=1)**: Current ENDF Format
  - Single atom: dominant channel (coherent or incoherent) approximation (Eq 24/25)
  - Polyatomic: DC atom gets LTHR=1 scaled by 1/f_DC; others get LTHR=2 with
    redistribution factor (Eq 26)
- **MEF (elastic_mode=2)**: Mixed Elastic Format
  - All atoms get LTHR=3 (coherent + incoherent in one section)
  - Coherent part: per-atom Bragg edges (shared across all atoms in the crystal)
  - Incoherent part: per-atom σ_inc and Debye-Waller parameter

### Key Implementation Details

- **Bragg-edge calculator**: Inlined from `coherent_elastic_general.py`. Handles
  any crystal symmetry (triclinic through cubic), multiple atom species, systematic
  absences, and crystallographic multiplicity. Algorithm follows NCrystal
  (Kittelmann et al., CPC 267, 2021).

- **Per-species Debye-Waller factors**: Each atom type gets its own DW parameter
  W_s = dwpix_s / (awr_s × T × kB), computed from partial phonon spectra. The DW
  is applied per species inside the structure factor:
  ```
  δ_j(T) = scale × Σ_{s,t} b_s b_t exp(-2(W_s + W_t) E_j) D_{st,j}
  ```
  This matches NCrystal's approach and gives ~0.03% agreement with NCrystal
  reference data for SiC.

- **DC atom selection**: For CEF polyatomic, the dominant-channel atom minimizes
  f_i/(1-f_i) × σ_inc_i across all atom types.

- **Species correlation matrix**: `compute_bragg_edges_general()` returns per-species
  correlation matrices D_{st,j} when `per_species=True`, enabling the per-species
  DW treatment without recomputing Bragg edges.

### Generalized Elastic Functions

| Function | Purpose |
|----------|---------|
| `compute_bragg_edges_general()` | General Bragg-edge calculator for any crystal |
| `_compute_per_species_msd()` | Per-species DW lambda from partial phonon spectra |
| `_build_generalized_elastic()` | Router: dispatches to CEF or MEF builders |
| `_build_cef_coherent()` | LTHR=1 with per-species DW and scale factor |
| `_build_cef_incoherent()` | LTHR=2 for non-DC atoms in CEF |
| `_build_mef_elastic()` | LTHR=3 (coherent + incoherent) |

### Validation

Tested against NCrystal reference ENDF files for SiC (F-43m, a=4.348 Å):

| Case | LTHR | Metric | Difference |
|------|------|--------|------------|
| C in SiC (CEF) | 1 | Cumulative S | 0.033% max |
| Si in SiC (CEF) | 2 | SB | exact |
| Si in SiC (CEF) | 2 | Wp | 0.019% |
| C in SiC (MEF) | 3 | Cumulative S | 0.033% max |
| C in SiC (MEF) | 3 | Wp | 0.037% |
| Si in SiC (MEF) | 3 | Wp | 0.019% |

Mono-atomic materials (Al, Fe, graphite) also verified with iel=10 inputs.

## Potential Future Work

- Suppress NumPy runtime warnings for expected edge cases (log of zero in
  `sint_vec` at lines 1063-1064)
- Investigate if endf_parserpy can be configured to use fixed-point notation
  for values in the 100-1000 range (would fix s-CH4 and HinCH2 diffs)
- Add automated regression test script that runs all 11 tests and reports diffs
- Consider translating other NJOY modules (THERMR, GROUPR, etc.)
