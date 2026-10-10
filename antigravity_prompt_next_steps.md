Before changing anything, do this first and show me the output:
Print the full contents of additive_descriptors.py and polymer_descriptors.py.
I need to see what feature engineering already exists there (e.g. RDKit
fingerprints, chemical-family encoding) before I ask you to duplicate work
that might already be done.

Then make the following changes, IN THIS ORDER. After EVERY numbered change,
stop and give me a short "CHANGED FILES" report: which file(s), which
function(s)/lines, a one-paragraph diff summary, and confirm nothing else in
that file was touched. Do not batch multiple numbered changes into one
report. Do not proceed to the next numbered change until I say continue.

============================================================
1. LOG-TRANSFORM water permeability (LMH/bar) specifically
============================================================
This target is right-skewed (mean ~185, max ~809 across the 318 rows). Add
an option to wrap the target in np.log1p before fitting and np.expm1 on
predictions before scoring, for water permeability ONLY -- leave every other
target untouched. Implement this as a wrapper/flag in evaluation.py
(repeated_kfold_eval and single_split_eval) rather than hacking the raw
dataframe, so R2/RMSE are still computed on the original (un-transformed)
scale for fair comparison to Gao's numbers -- i.e. fit on log1p(y), predict,
then expm1 the predictions before computing r2_score/rmse against the
original y_test. Do NOT change the metric scale, only the space the model
trains in.

============================================================
2. INCREASE Optuna trial budget substantially
============================================================
Current budgets (N_TRIALS_GBM / N_TRIALS_OTHER in run_dataset_a.py,
run_dataset_b.py, run_dataset_c.py, and wherever tuning.py defines defaults)
are far too thin -- tuning finishes in 5-90 seconds, meaning ~6-8 trials.
Bump to:
  - LightGBM / XGBoost_reproduction: 60 trials
  - HistGB: 50 trials
  - ExtraTrees: 40 trials
  - SVR: 40 trials
Keep the existing inner-CV nested tuning structure (tune only on the outer
training fold, never touch the outer test/val fold) -- do not change the
leakage-safety logic, only the n_trials numbers. Tell me the new expected
runtime per (target, model) pair based on the per-trial time you're already
seeing in the existing logs, so I know what I'm signing up for before running
it overnight again.

============================================================
3. ADD a clean weighted-average ensemble (NOT stacking, NOT RidgeCV)
============================================================
For each target, after the tuned single models are done: take the 2-3 best
models by repeated-CV R2 mean, and build a simple ensemble that averages
their predictions weighted by each model's inverse CV RMSE (better model =
higher weight), computed OUT-OF-FOLD (weights determined on validation
folds, never on the final held-out test fold, same leakage discipline as
everywhere else). No meta-learner, no RidgeCV, no internal StackingRegressor
cv split -- just a weighted average, evaluated with the SAME
repeated_kfold_eval / single_split_eval harness as everything else so
numbers are directly comparable. Add it as a new model name
"WeightedEnsemble" alongside the existing ones in MODEL_NAMES for
run_dataset_a.py and run_dataset_b.py only (skip Dataset C for now).

============================================================
STOP HERE and wait for me
============================================================
Do not start the two-stage pipeline (Sheet 2 properties -> Sheet 1
performance) yet -- that's a bigger change I'll scope separately after I see
whether 1-3 close the gap enough on their own.
