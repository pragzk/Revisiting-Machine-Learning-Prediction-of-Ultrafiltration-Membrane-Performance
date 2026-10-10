# Data Audit Report — Gao et al. (2023) UF Membrane Dataset
Source: `es2c05404_si_002.xlsx`. All numbers below are from executed pandas code, not from the paper text.

## 1. Sheet structure (as actually found — differs from a naive read)

| Sheet | Real header row | Rows (non-empty) | Role |
|---|---|---|---|
| `1` | Excel row 3 (2 rows of group labels above it) | **318** | Framework A: fabrication + operational conditions → 5 performance targets |
| `2` | Excel row 2 (1 group-label row above it) | **301** | Stage-1 (two-stage pipeline): fabrication conditions → 3 membrane properties |
| `3` | Excel row 2; DOI column was unlabeled (`Unnamed: 13`), renamed | **328** | Stage-2 raw inputs: membrane properties + operational conditions → performance |

**None of the three sheets has exactly 320 rows.** The paper's "320" is presumably the size of the master compiled dataset before per-model missing-data filtering; each sheet here is a different filtered/derived view, not the same 320 rows three times.

## 2. Sheet 1 ↔ Sheet 3 relationship — checked, not assumed
Matching sheet 1 and sheet 3 rows on the 3 performance columns they share (water permeability, removal %, reversible fouling ratio) plus DOI gives only **231/318 exact matches** (and sheet 3 has 328 rows total, more than sheet 1's 318). So sheet 3 is **not** a strict superset or a re-ordering of sheet 1 — it's an independently filtered compilation (likely: rows kept if properties+operational+performance are present, regardless of whether full fabrication detail is present). This matters for the two-stage pipeline: Stage-1 outputs (trained on sheet 2's 301 rows) cannot be naively joined onto sheet 3's 328 rows by row position — needs a join key (DOI + fabrication conditions) or must be re-derived from a common merged table. Flagging this now before building Framework B so we don't silently leak or misalign.

## 3. Missingness (Sheet 1, n=318)
| Feature | % missing |
|---|---|
| P_MW (polymer MW) | 37.4% |
| pore maker molecular weight (Da) | 31.1% |
| surface tension (mN/m) | 20.1% |
| organic compound size (Da) | 19.5% |
| solubility parameter | 1.6% |
| everything else (incl. all 6 targets) | 0.0% |

Sheet 2 (properties): polymer SMILES 49.2% missing, polymer MW 47.2%, pore-maker MW 40.2%, surface tension 25.9%, intrinsic viscosity 12.6%, **DOI itself 4.3% missing** (13 rows with no source — will need to exclude these from DOI-grouped CV / LOSO, they can't be grouped).

## 4. Duplicates / data-quality issues found
- **No exact duplicate rows** in sheet 1.
- Categorical spelling inconsistencies that inflate cardinality artificially: `organic solvent` has `'DMAc'` (46) and `'DMAC'` (4) as separate categories — same solvent, will be canonicalized. Similarly expect `'Safranine O'` vs `'Safranin O'` in `organic compound` (13 nominal categories, likely 12 real).
- `additive` has 72 raw categories, many are numbered variants of the same base material (e.g., `GO-1`...`GO-9`, `TiO2 NPs-1`...`-5`, `ZIF-8` vs `ZIF-8 (Z1)` vs `ZIF-8 (Z2)`) — this is exactly the raw-categorical-ID problem section 5A asked to fix with a chemical-family layer (graphene-family, MOF, metal-oxide, CNT, etc.).

## 5. Target distributions (Sheet 1, n=318)
| Target | mean | std | min | max |
|---|---|---|---|---|
| Water permeability (LMH/bar) | 185.1 | — | — | 809.06 |
| Removal efficiency (%) | 87.5 | — | — | 99.59 |
| Flux decline ratio (%) | 42.8 | — | — | 83.5 |
| Flux recovery ratio (%) | 73.9 | — | — | 91.3 |
| Reversible fouling ratio (%) | 31.6 | — | — | 82.5 |
| Irreversible fouling ratio (%) | 26.2 | — | — | 73.0 |

**Confirmed exactly as the paper states**: `flux recovery ratio` and `irreversible fouling ratio` have Pearson r = **−1.000** (they're algebraic complements by construction, eq. 9/11 in the paper) — so irreversible fouling is correctly dropped as a redundant target, flux recovery kept as representative. Water permeability is right-skewed (max 809 vs mean 185) — will need to check if log-transform helps.

## 6. DOI grouping (critical for leakage-safe CV)
Sheet 1: **53 unique DOI groups**, group sizes range 1–25, median 5, mean 6.0. This is enough groups for GroupKFold (planned: 5-fold grouped) and for LOSO (53 folds — expensive but feasible; will time-box it).

## 7. Data dictionary (Sheet 1 — primary performance dataset)
| Column | Type | Role | Notes |
|---|---|---|---|
| Polymer | categorical (11 levels) | fabrication | base polymer |
| SMILES | string | fabrication | repeat-unit SMILES, 0% missing — usable for RDKit descriptors |
| P_MW | numeric | fabrication | 37% missing |
| P_wt% | numeric | fabrication | |
| surface tension (mN/m) | numeric | fabrication | 20% missing |
| pore maker | categorical (4: PEG/none/PVP/Pluronic F127) | fabrication | |
| pore maker MW (Da) | numeric | fabrication | 31% missing |
| pore maker wt% | numeric | fabrication | |
| additive | categorical (72 raw levels) | fabrication | needs chemical-family encoding |
| additive wt% | numeric | fabrication | |
| organic solvent | categorical (5 raw / 4 real) | fabrication | DMAc/DMAC dupe |
| solubility parameter | numeric | fabrication | 1.6% missing |
| TMP (bar) | numeric | operational | transmembrane pressure |
| organic compound | categorical (13 raw) | operational | test solute (BSA, HA, dyes...) |
| organic compound conc. (mg/L) | numeric | operational | |
| organic compound size (Da) | numeric | operational | 19.5% missing |
| foulant | categorical (4) | operational | |
| foulant conc. (mg/L) | numeric | operational | |
| DOI | string | grouping key | source paper — use for GroupKFold/LOSO |
| **6 targets** | numeric | outputs | see §5 |

Full column-level stats saved to `results/01_data_dictionary.csv`.
