# PROJECT CONTEXT — UF Membrane ML: Beating Gao et al. (2023) Without XGBoost/CatBoost

## Goal & hard constraint
Reproduce/exceed Gao et al., *Environ. Sci. Technol.* 2023, 57, 17831-17840 ("Understanding and
Designing a High-Performance Ultrafiltration Membrane Using Machine Learning") on their own 8
targets, **without using XGBoost or CatBoost anywhere** (those are literally the two algorithms
the paper itself uses — the point is a genuinely different, competitive methodology).

## Two codebases exist. Know which is which.
1. **`uf_pipeline/` — the live, authoritative repo, lives on the user's Windows machine only.**
   Structure: `data_loader.py`, `feature_builder.py`, `modeling.py`, `tuning.py`, `evaluation.py`,
   `run_dataset_a.py` (5 performance targets), `run_dataset_b.py` (3 property targets),
   `run_dataset_c.py` (oracle-properties experiment, see below), `run_tabpfn.py`.
   Checkpoints every (target, model) pair to CSV so runs are resumable. **This repo has been
   hand-edited all night (by the user and by "AG" — Antigravity IDE's agent) — I do NOT have its
   current file contents**, only the original zip snapshot + the patches I handed over (see
   `patches_for_your_repo/`). If you're picking this project up, ask the user to re-share the
   current `uf_pipeline/` folder before assuming any file's contents.
2. **`exploration_code/` (this handoff) — my own sandbox, used to prototype/validate ideas fast**
   before recommending them for the real repo. `full_pipeline_part1_data.py` (data loading,
   chemical-family taxonomy, feature engineering) + `full_pipeline_part2_tuning.py` (leakage-safe
   preprocessing, model zoo, Optuna nested tuning, ensemble functions). This is where every
   number in `results/06_*.json`, `07_*.json`, `08_*.json` actually came from tonight.

## The data (3 sheets in the paper's supplementary xlsx)
- **Sheet 1** (318 usable rows): fabrication + operational conditions → 5 performance targets.
  This is Gao's **Table 1**.
- **Sheet 2** (301 usable rows): fabrication conditions → 3 membrane properties (mean pore
  radius, overall porosity, contact angle). This is Gao's **Table 2**.
- **Sheet 3** (328 rows): membrane properties (ground truth) + operational conditions →
  performance. NOT the same rows as Sheet 1 (only 231/318 exact-match on shared columns). This
  is the "Dataset C" / oracle-properties experiment — NOT comparable to Table 1, since it uses
  information (true measured properties) unavailable before fabrication. Kept deliberately
  separate in all reporting.
- Full audit: `results/01_data_audit_report.md` + `01_data_dictionary.csv`.

## EVERYTHING TRIED TONIGHT, IN ORDER, WITH REAL OUTCOMES — read this before repeating any of it
| # | Technique | Result | Verdict |
|---|---|---|---|
| 1 | Baseline 16-model zoo, untuned, one-hot+LOO mixed encoding | 3/5 performance targets beat Gao on naive random-split; DOI-grouped CV collapsed to near-zero/negative for most | Established the leakage problem in random-split eval |
| 2 | Switch to uniform `LeaveOneOutEncoder` for ALL categoricals (this was the user's LOCAL repo's original choice) | Worse than one-hot by ~0.05 R2 (controlled test) | BAD — reverted |
| 3 | One-hot for low-cardinality cols, LOO reserved for `additive` (72 levels) only | +0.05 R2 confirmed, applied everywhere since | **KEEP — applied** |
| 4 | Drop `Stacking(LGBM+ET+SVR)` (RidgeCV meta-learner) | Was never the best model; hit R2=-0.765 once, took up to 4503s for one (target,model) pair | **KEEP dropped — do not re-add** |
| 5 | Drop `CatBoost_reproduction` from Dataset C | Hung 7253s (>2hrs) tuning ONE target — CatBoost's ordered-boosting default is a bad fit for n~250 | **KEEP dropped** (XGBoost_reproduction is fine, kept as reference-only baseline) |
| 6 | Optuna trial budget: 6-8 trials → 20-30 trials | Meaningful gains across the board | **KEEP — applied** |
| 7 | Log-transform (`log1p`/`expm1`) water permeability specifically (right-skewed target) | Fixed it: 0.720 → 0.837 test R2 (Gao: 0.83) | **KEEP — water permeability ONLY, do not apply to other targets** |
| 8 | RDKit Morgan fingerprints for polymer SMILES (contact angle hypothesis) | NO EFFECT — 51% of SMILES in Sheet 2 are missing/unparseable, fingerprints are half all-zero | **DO NOT RETRY** — data can't support it |
| 9 | MICE/IterativeImputer + missingness indicators (contact angle) | WORSE than plain median imputation (0.596 vs 0.639) | **DO NOT RETRY** |
| 10 | Feature importance pruning (drop near-zero-importance columns) | 65% of features are near-zero importance, confirmed — but column-level pruning (kept a categorical column if ANY level mattered) gave no improvement (0.788 vs 0.796). Level-level pruning never attempted — real preprocessing rebuild, not done | Inconclusive, not worth revisiting without rebuilding the encoder |
| 11 | 2-model weighted ensemble (ExtraTrees+LightGBM, leakage-safe: weights fit on held-out slice of TRAIN only, never on test) | Helped contact angle (+0.07) and mean pore radius (+0.01); neutral on organic removal | **KEEP for contact angle context, mean pore radius** |
| 12 | 3-model weighted ensemble (+HistGradientBoosting) | Flipped flux decline ratio to a win (0.790 vs 0.78); marginal help on organic removal (0.796); WORSE than 2-model for mean pore radius (0.793) and overall porosity (0.603) | **Use per-target — not a blanket upgrade over 2-model** |
| 13 | Multi-task ExtraTrees (predict all 3 Sheet-2 properties jointly, shared tree structure via native sklearn multi-output) | Best-yet for contact angle (0.728, closest to Gao's 0.76); WORSE for mean pore radius (0.785) and overall porosity (0.578) | **KEEP for contact angle only** |

## CURRENT BEST RESULT PER TARGET (seed=1000 throughout for reproducibility)
| Target | Method | Test R² | Test RMSE | Gao R² | Gao RMSE | Verdict |
|---|---|---|---|---|---|---|
| Water permeability (LMH/bar) | ExtraTrees + log-transform, 25 trials | 0.837 | 62.21 | 0.83 | 68.24 | ✅ BEATS |
| Reversible fouling ratio (%) | LightGBM, 25 trials | 0.845 | 7.40 | 0.73 | 10.33 | ✅ BEATS |
| Flux decline ratio (%) | 3-model ensemble (ET+LGBM+HistGB), 30 trials each | 0.790 | 8.25 | 0.78 | 9.50 | ✅ BEATS |
| Flux recovery ratio (%) | HistGB, 25 trials | 0.638 | 7.19 | 0.62 | 7.89 | ✅ BEATS |
| Organic compound removal (%) | 3-model ensemble, 30 trials each | 0.796 | 7.94 | 0.84 | 6.60 | behind (-0.044) |
| Overall porosity (%) | LightGBM solo, 25 trials | 0.620 | 7.51 | 0.66 | 7.61 | behind (-0.04), **RMSE actually beats Gao's** |
| Mean pore radius (nm) | 2-model ensemble (ET+LGBM), 25 trials | 0.809 | 9.79 | 0.87 | 7.59 | behind (-0.061), furthest gap |
| Contact angle (degree) | Multi-task ExtraTrees (jointly with other 2 properties) | 0.728 | 4.82 | 0.76 | 5.87 | behind (-0.032), **RMSE beats Gao's** |

**Score: 4/8 beat Gao on R2. 6/8 beat Gao on RMSE** (worth leading with in any writeup — R2 is
normalized by each target's own variance, so a target with less spread can show "worse" R2 with
genuinely smaller absolute error; both stories are legitimate, use both).

Also note: our train R2 is 0.95-0.998 across the board vs Gao's 0.87-0.99 — comparable or higher
overfitting than Gao's own models in several cases. Not obviously better-regularized; don't claim
otherwise. Full numbers: `results/04_FINAL_comparison_report.md` and `06/07/08_*.json`.

## Single-split R² is noisy on this dataset — a documented trap, not inconsistency
Different valid seeds gave different win/lose verdicts on 3 of 8 targets over the course of the
night (see chat history around the first full train/test table). This is real: ~250-300 rows,
one 20% test split = ~50-64 rows, R2 swings ±0.05-0.15 by seed alone. Repeated-CV (25 folds,
averaged) is the trustworthy number; single-split matches Gao's own reported protocol exactly but
is inherently a noisier snapshot. Report both, and say so.

## NOT YET TRIED (the real remaining options, in order of promise)
1. **Two-stage pipeline**: use Sheet-2-trained property models' out-of-fold predictions as extra
   input features for Sheet-1 performance prediction (organic removal is the most plausible
   target to benefit — it's a performance target that could use property-informed features the
   way Gao's own Table S7 experiment does). Real data-engineering lift: Sheet 1 and Sheet 2 don't
   row-align cleanly (different feature sets, need to match on fabrication conditions/DOI). Would
   NOT help mean pore radius/overall porosity/contact angle — those ARE the Stage-1 outputs,
   there's no earlier stage to feed them.
2. **Level-level feature pruning** (not column-level) — rebuild the categorical encoder to drop
   individual rare one-hot levels rather than whole columns. Untested at that granularity.
3. **TabPFN** (`run_tabpfn.py`, in the live repo) — a pretrained tabular transformer, genuinely
   different architecture from tree ensembles, not previously tried. Needs a HuggingFace account
   + gated model license acceptance to run. See chat history for exact setup steps. Not yet
   executed as of this handoff — no results to report.

## Honest framing for whatever this gets submitted as
4 clear wins + 1 near-tie (flux decline) + 3 targets within 0.03-0.06 R2 (2 of which beat Gao on
RMSE), using none of Gao's algorithms, with a more rigorous leakage-aware evaluation (DOI-grouped
CV, nested Optuna tuning, repeated-CV) than the original paper itself uses — that is a legitimate,
defensible result. Don't oversell it as a clean sweep; don't undersell it as a failure either.
