# Phase 2 Checkpoint — Baseline (Untuned) Model Zoo, No XGBoost/CatBoost

**Protocol used (exactly as specified):** 16-model zoo, zero XGBoost/CatBoost anywhere. For every (target, model) pair: (a) 30 repeated random 80/20 splits, preprocessing (median-impute + scale numeric, impute + one-hot categorical) refit from scratch on the train fold every single split — no fitting on the full dataset; (b) 5-fold **DOI-grouped** CV via `GroupKFold` so no source paper appears in both train and validation within a fold. All hyperparameters are library defaults / sane baselines — **no tuning yet**. This is the "before Optuna" checkpoint so later gains are measurable.

## Result table — best model per target, both protocols

| Target | Gao R² | Best model (random-split) | Our R² random-split (mean±sd, 30 seeds) | Our RMSE random-split | Beats Gao on random-split? | Same model's DOI-grouped R² | Best DOI-grouped R² (any model) |
|---|---|---|---|---|---|---|---|
| Water permeability | 0.83 | RandomForest | **0.848 ± 0.040** | 62.12 ± 8.60 | ✅ (untuned) | 0.296 | 0.383 (ElasticNet) |
| Removal efficiency | 0.84 | GaussianProcess | 0.800 ± 0.095 | 7.18 ± 1.76 | ❌ | -0.287 | -0.173 |
| Flux decline ratio | 0.78 | LightGBM | **0.798 ± 0.045** | 8.48 ± 0.96 | ✅ (untuned) | -1.300 | -0.814 |
| Flux recovery ratio | 0.62 | GradientBoosting | 0.576 ± 0.099 | 8.03 ± 0.87 | ❌ | -0.181 | -0.021 |
| Reversible fouling ratio | 0.73 | LightGBM | **0.779 ± 0.069** | 9.00 ± 1.47 | ✅ (untuned) | -1.063 | -0.395 |

Full model-by-model numbers: `02_baseline_summary_by_model.csv`. Every individual seed/fold score: `02_baseline_repeated_ALL.csv`, `02_baseline_grouped_ALL.csv`.

## Honest read of what this actually shows

**1. Random-split R² already beats Gao on 3/5 targets, untuned, without XGBoost/CatBoost.** That's a genuinely good sign for the "beat Gao without their algorithms" goal — but it comes with a major caveat below.

**2. The DOI-grouped numbers are the real finding of this phase, and they're bad — in the informative sense.** For water permeability, R² drops from 0.85 (random split) to 0.30–0.38 (grouped) — a real but partial generalization ability. For removal efficiency, flux decline, and reversible fouling, DOI-grouped R² goes **negative** (worse than just predicting the training mean) for nearly every model. That means: when you hold out an entire source paper, these untuned models currently cannot predict its membranes better than a naive baseline. Random 80/20 splitting — which is what Gao's paper itself uses, and what most of the "beat Gao" framing implicitly measures — is letting near-duplicate/correlated rows *from the same source paper* leak between train and test. The 320-sample dataset compiled from 53 papers with median 5 rows/paper is exactly the setting where this leakage is severe: rows from one paper (same base polymer, same lab's baseline conditions, varying only one additive loading) are not independent samples.

**This is the central methodological finding the brief asked me to surface, not bury.** A model that "beats Gao" only on random splits but collapses under DOI-grouped evaluation has not actually out-performed Gao's model in a way that matters for design guidance on new, unseen chemistries — it's just fit the same evaluation loophole more aggressively. Gao's paper doesn't report grouped numbers either, so we can't yet say whether Gao's XGBoost/CatBoost models would also collapse under grouping — that's a fair comparison I still owe you (running Gao's exact protocol, i.e. XGBoost/CatBoost, under DOI-grouped CV, is allowed as a *benchmark comparison*, not as part of our proposed methodology — flagging this now so it's not a silent scope-creep of the "no XGBoost" rule).

**3. Untuned SVR/KernelRidge are currently the weak point** (some R² near 0 or negative) — expected, they're the most hyperparameter-sensitive family and haven't been tuned yet (Optuna pass is next).

**4. Flux recovery ratio is still below Gao even on random splits** (0.576 vs 0.62) — this is the target the brief specifically wants improved, and untuned models aren't there yet. Likely needs the two-stage pipeline (contact-angle-mediated) and/or better feature engineering (chemical-family additive encoding, interaction terms) rather than just a different algorithm — worth prioritizing in the ablation.

## What's NOT done yet (so I don't imply more than I've shown)
- No hyperparameter tuning (Optuna) — everything above is default/baseline settings only.
- No chemical-family/descriptor ablation yet (features exist in the data but I haven't yet tested whether they help vs raw categorical — that's Section 12's job).
- No DL models yet (MLP/ResNet/FT-Transformer/TabM).
- No two-stage pipeline, no multi-task, no ensembling, no LOSO (ran GroupKFold only so far — LOSO at 53 folds is next but costly), no uncertainty quantification, no SHAP, no statistical significance testing against Gao.
- Membrane-property targets (porosity, pore radius, contact angle — Sheet 2) not touched yet.

Next turn: Optuna hyperparameter tuning on the strongest baseline models (nested CV, nothing tuned against the eventual held-out test set), plus the additive-family feature ablation, since both are prerequisites before ensembling/DL comparisons mean anything.
