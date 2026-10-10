# How to Run This Yourself

## 0. What you need to decide first
There are two separate codebases in your project (per `CONTEXT.md`). This guide
covers **`exploration_code/`** (the one that actually produced the numbers in
your paper: `06_train_test_batch1.json`, `07_ensemble_batch1.json`,
`08_ensemble_train_metrics.json`) since that's the one I have real content for.

If you instead meant `uf_pipeline/` (the `run_dataset_a/b/c.py` +
`data_loader/feature_builder/modeling/tuning/evaluation.py` repo on your
Windows machine) — I only have `run_dataset_c.py` and `build_final_table.py`
from that repo. You'd need to share the other 5 files before I can give exact
commands for it; the steps below (env setup, getting the data file) still
apply either way.

## 1. Environment setup

```bash
python3 -m venv uf_env
source uf_env/bin/activate        # Windows: uf_env\Scripts\activate

pip install pandas numpy scikit-learn lightgbm optuna category_encoders openpyxl
# Only needed if you're re-running the SMILES fingerprint experiment:
pip install rdkit
```

Notes:
- `openpyxl` is required for `pandas.read_excel` to read `.xlsx` — without it
  `load_sheet1()`/`load_sheet2()`/`load_sheet3()` will fail.
- No XGBoost/CatBoost install needed for the proposed methodology itself
  (only `run_dataset_c.py` references them, as reference-only reproduction
  baselines — and your own notes say CatBoost hung for 2+ hours on this data
  size, so you may want to skip installing it and just accept `run_dataset_c.py`
  will error/skip on that one model).

## 2. Get the actual data file

Neither of us has `es2c05404_si_002.xlsx` — it's Gao et al.'s Supporting
Information file, not something either codebase ships with. Download it from
the paper's ACS page:

https://pubs.acs.org/doi/10.1021/acs.est.2c05404

(Supporting Information link near the top — it's the `.xlsx` file referenced
throughout your audit report.) Place it at:

```
<your project root>/data/es2c05404_si_002.xlsx
```

matching the `DATA_PATH = 'data/es2c05404_si_002.xlsx'` constant in
`full_pipeline_part1_data.py`. If you run the script from a different working
directory, either `cd` into the folder containing `data/`, or edit
`DATA_PATH` to an absolute path.

## 3. Run Part 1 — data loading + feature engineering

```bash
python3 full_pipeline_part1_data.py
```

This is self-contained and safe to run as-is — its `if __name__ == '__main__'`
block loads all 3 sheets, applies the chemical-family taxonomy and interaction
features, and writes:

```
outputs/sheet1_engineered.pkl
outputs/sheet2_engineered.pkl
outputs/sheet3_engineered.pkl
```

It prints the resulting shapes and the list of newly added columns, so you'll
immediately see if the additive-family mapping picked up unmapped categories
(anything not in `ADDITIVE_FAMILY` falls back to `'unknown'` — worth checking
that list is empty or small on your actual file, since the mapping was built
by hand against the audit report's category list).

The `SMOKE_TEST` env var only controls seed/trial/epoch counts, which are used
by Part 2 downstream, not by anything in Part 1's own logic — but Part 1 does
print the resolved config, so set it before running either script:

```bash
export UF_SMOKE_TEST=1   # fast, few seeds/trials — sanity-check first
python3 full_pipeline_part1_data.py
python3 full_pipeline_part2_tuning.py   # <- see caveat below

export UF_SMOKE_TEST=0   # full 30-seed, full Optuna budget run — slow
python3 full_pipeline_part1_data.py
python3 full_pipeline_part2_tuning.py
```

## 4. Run Part 2 — tuning, ensembling, evaluation

Now that I have the actual content of `full_pipeline_part2_tuning.py`, here's
the real situation: **this file has no CLI and no driver that produces
`06/07/08_*.json` or `02/09_baseline_*.csv`.** Its only `if __name__ ==
'__main__':` block is a 2-seed, 5-trial smoke test on one target
(`water permeability`, `ExtraTrees`) — useful for confirming things import
and run, but it's not what generated your reported numbers. Whatever script
actually looped over all 8 targets and dumped those JSON/CSV files isn't
among anything you've shared.

I've written that missing driver for you: **`run_full_pipeline.py`** (in the
outputs panel). It calls the real functions defined in your part2 file —
`tune_with_train_test_metrics()` (produces the exact shape of
`06_train_test_batch1.json`), `build_and_eval_ensemble_n_with_train()`
(produces `08_ensemble_train_metrics.json`), and
`evaluate_model_repeated_splits()` / `evaluate_model_group_kfold()` across the
full 16-model `BASELINE_MODEL_ZOO` (produces `02/09_baseline_*.csv`).

Place it in the same folder as `full_pipeline_part1_data.py` and
`full_pipeline_part2_tuning.py`, then:

```bash
# 1. Sanity check first -- ~1 minute, confirms data loads and tuning runs end to end
python3 run_full_pipeline.py --stage smoke

# 2. Baseline 16-model screen (this is the slow one -- 30 seeds x 16 models x
#    5 performance targets, plus 15 seeds x 16 models x 3 property targets,
#    plus DOI-grouped 5-fold CV for the performance targets)
python3 run_full_pipeline.py --stage baseline

# 3. Optuna-tuned single models, seed=1000, all 8 targets
python3 run_full_pipeline.py --stage tuned --n-trials 25

# 4. Weighted ensembles for the targets your notes say benefited from them
python3 run_full_pipeline.py --stage ensemble --n-trials 25

# Or all four in sequence:
python3 run_full_pipeline.py --stage all --n-trials 25
```

Outputs land in `outputs/`, matching your original filenames
(`02_baseline_repeated_ALL.csv`, `06_train_test_batch1.json`, etc.) so you can
diff them directly against what you already have.

**Two things I had to make editorial calls on — check these against your own
notes before trusting the output:**
1. **Which models tuning actually supports.** Only 9 of the 16
   `BASELINE_MODEL_ZOO` models have `suggest_params()`/`make_model_from_params()`
   defined in your file (`TUNABLE_MODELS = ['ExtraTrees', 'RandomForest',
   'HistGradientBoosting', 'LightGBM', 'SVR_rbf', 'GradientBoosting', 'Ridge',
   'ElasticNet', 'KernelRidge_rbf']`). AdaBoost, SVR_poly, Lasso, PLS, KNN,
   BayesianRidge, and GaussianProcess — including the GradientBoosting-adjacent
   models your baseline screen flagged as strong — can only ever run untuned
   in this codebase as it stands. The "tuned" stage in my driver picks the
   best test R² across only those 9.
2. **Which models go into each target's ensemble.** Your report files
   describe *what was used* (e.g. "3-model ensemble ET+LGBM+HistGB" for flux
   decline ratio) but the actual ensemble-membership logic wasn't in any file
   you shared. I hardcoded `ENSEMBLE_MEMBERS` in the driver to match your
   README's stated combinations — verify this against your own notes,
   especially for overall porosity and contact angle, where your files
   disagree with each other about which models/method actually won (06 json
   says ExtraTrees solo for overall porosity; your paper text and README say
   LightGBM solo was the reported "best"; 08 json's ensemble entry for overall
   porosity has no `weights` field, unlike every other target, suggesting it
   may not be a genuine multi-model ensemble at all in the original run).

## 5. Building the final comparison table

Once results exist on disk, `build_final_table.py` is safe to re-run any time
(read-only, never re-fits anything):

```bash
python3 build_final_table.py
```

This expects `results/dataset_a_repeated_cv.csv`, `dataset_a_single_split.csv`
(and the `_b`/`_c` equivalents) to exist. **This is a `uf_pipeline/`-style
script** (it imports nothing but `pandas`/`pathlib`, so it'll run standalone,
but the CSV naming convention it expects — `dataset_a_repeated_cv.csv` etc. —
matches `run_dataset_a.py`/`run_dataset_b.py`/`run_dataset_c.py`'s output
convention, not `full_pipeline_part2_tuning.py`'s). If your actual results are
sitting in `06/07/08_*.json` instead (the exploration_code naming), this
script will just print "No Dataset A/B results on disk yet." — you'd need to
either convert those JSONs to the expected CSV shape, or write a small
adapter script (I can write that for you if you confirm this is the situation
you're in).

## 6. If you're running `uf_pipeline/` instead

`run_dataset_c.py`'s pattern (which I do have in full) is:

```bash
python3 run_dataset_c.py --target-idx 0                      # all models, one target
python3 run_dataset_c.py --target-idx 0 --model LightGBM      # one model, one target
python3 run_dataset_c.py --target-idx 0 --quick                # fewer repeats, fast sanity check
python3 run_dataset_c.py --summary                             # print results from disk, no fitting
```

`run_dataset_a.py` (5 performance targets) and `run_dataset_b.py` (3 property
targets) almost certainly follow the same `--target-idx` / `--model` /
`--quick` / `--summary` pattern, based on shared imports
(`data_loader`, `feature_builder`, `modeling`, `tuning`, `evaluation`) and
identical checkpoint-per-pair design described in `CONTEXT.md` — but **I'm
inferring this from `run_dataset_c.py`'s pattern, not confirming it**, since I
haven't seen those two files. Check their own `argparse` block before running.

This codebase also needs `data_loader.py` to expose a `load_dataset_a()` /
`load_dataset_b()` / `load_dataset_c()` function (used exactly this way in
`run_dataset_c.py`) pointing at the same `es2c05404_si_002.xlsx` from step 2.

## Fastest path if you just want to sanity-check something works today

```bash
export UF_SMOKE_TEST=1
python3 full_pipeline_part1_data.py
```

This alone will tell you within seconds whether your data file path is
correct and whether the chemical-family mapping covers your actual additive
categories — before you commit to any multi-hour tuning run.
