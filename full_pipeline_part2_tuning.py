"""
Part 2: leakage-safe preprocessing + evaluation harness + classical ML model
zoo + Optuna nested-CV hyperparameter tuning.

ABSOLUTELY NO XGBoost. ABSOLUTELY NO CatBoost. Anywhere in this file.

Leakage rule enforced everywhere: every fit-able transform (imputer, scaler,
one-hot encoder) is `.fit()` on the TRAIN fold only inside each split/fold
function, never on the full dataset beforehand.
"""
import numpy as np
import pandas as pd
import warnings, time
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                               HistGradientBoostingRegressor, GradientBoostingRegressor,
                               AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# --------------------------------------------------------------------------
# Leakage-safe preprocessing (unfitted -- caller fits on train fold only)
# --------------------------------------------------------------------------
def build_preprocessor(numeric_cols, categorical_cols, impute_strategy='median'):
    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy=impute_strategy)),
        ('scale', StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', numeric_pipe, numeric_cols),
        ('cat', categorical_pipe, categorical_cols),
    ], remainder='drop')


# --------------------------------------------------------------------------
# NO XGBoost. NO CatBoost. Baseline (default-hyperparameter) model zoo.
# --------------------------------------------------------------------------
BASELINE_MODEL_ZOO = {
    'ExtraTrees': lambda: ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    'RandomForest': lambda: RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    'HistGradientBoosting': lambda: HistGradientBoostingRegressor(random_state=42, max_iter=300),
    'GradientBoosting': lambda: GradientBoostingRegressor(random_state=42, n_estimators=300),
    'AdaBoost': lambda: AdaBoostRegressor(random_state=42, n_estimators=200),
    'LightGBM': lambda: lgb.LGBMRegressor(random_state=42, n_estimators=400, verbosity=-1),
    'SVR_rbf': lambda: SVR(kernel='rbf', C=10, gamma='scale'),
    'SVR_poly': lambda: SVR(kernel='poly', C=10, degree=3),
    'Ridge': lambda: Ridge(alpha=1.0, random_state=42),
    'ElasticNet': lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000),
    'Lasso': lambda: Lasso(alpha=0.1, random_state=42, max_iter=5000),
    'PLS': lambda: PLSRegression(n_components=8),
    'KNN': lambda: KNeighborsRegressor(n_neighbors=7),
    'BayesianRidge': lambda: BayesianRidge(),
    'KernelRidge_rbf': lambda: KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1),
    'GaussianProcess': lambda: GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        random_state=42, normalize_y=True, n_restarts_optimizer=2),
}


def evaluate_model_repeated_splits(model_factory, X, y, numeric_cols, categorical_cols,
                                    seeds, test_size=0.2):
    records = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        pre = build_preprocessor(numeric_cols, categorical_cols)
        Xtr = pre.fit_transform(X_train)
        Xte = pre.transform(X_test)
        model = model_factory()
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        records.append({'seed': seed, 'r2': r2_score(y_test, pred), 'rmse': rmse(y_test, pred),
                         'mae': mean_absolute_error(y_test, pred), 'n_train': len(y_train), 'n_test': len(y_test)})
    return pd.DataFrame(records)


def evaluate_model_group_kfold(model_factory, X, y, groups, numeric_cols, categorical_cols, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    records = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pre = build_preprocessor(numeric_cols, categorical_cols)
        Xtr = pre.fit_transform(X_train)
        Xva = pre.transform(X_val)
        model = model_factory()
        model.fit(Xtr, y_train)
        pred = model.predict(Xva)
        records.append({'fold': fold, 'r2': r2_score(y_val, pred), 'rmse': rmse(y_val, pred),
                         'mae': mean_absolute_error(y_val, pred), 'n_train': len(y_train), 'n_val': len(y_val)})
    return pd.DataFrame(records)


def evaluate_model_loso(model_factory, X, y, groups, numeric_cols, categorical_cols, max_folds=None):
    logo = LeaveOneGroupOut()
    all_splits = list(logo.split(X, y, groups))
    if max_folds is not None and len(all_splits) > max_folds:
        val_sizes = [len(v) for _, v in all_splits]
        order = np.argsort(val_sizes)[::-1][:max_folds]
        all_splits = [all_splits[i] for i in order]
    records = []
    for train_idx, val_idx in all_splits:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        held_out_doi = groups.iloc[val_idx].iloc[0]
        pre = build_preprocessor(numeric_cols, categorical_cols)
        Xtr = pre.fit_transform(X_train)
        Xva = pre.transform(X_val)
        model = model_factory()
        model.fit(Xtr, y_train)
        pred = model.predict(Xva)
        rec = {'held_out_doi': held_out_doi, 'n_val': len(y_val), 'n_train': len(y_train)}
        rec['r2'] = r2_score(y_val, pred) if len(y_val) > 1 else np.nan
        rec['rmse'] = rmse(y_val, pred)
        rec['mae'] = mean_absolute_error(y_val, pred)
        records.append(rec)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# SECTION 11: OPTUNA HYPERPARAMETER OPTIMIZATION
# Search spaces tuned INSIDE a train-only inner KFold (nested CV): the
# objective function for each trial does its own inner K-fold split of the
# outer-training data, fits on inner-train, scores on inner-val. The outer
# test/val fold is NEVER seen during tuning.
# --------------------------------------------------------------------------
def make_model_from_params(model_name, params):
    if model_name == 'ExtraTrees':
        return ExtraTreesRegressor(random_state=42, n_jobs=-1, **params)
    if model_name == 'RandomForest':
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if model_name == 'HistGradientBoosting':
        return HistGradientBoostingRegressor(random_state=42, **params)
    if model_name == 'LightGBM':
        return lgb.LGBMRegressor(random_state=42, verbosity=-1, **params)
    if model_name == 'SVR_rbf':
        return SVR(kernel='rbf', **params)
    if model_name == 'GradientBoosting':
        return GradientBoostingRegressor(random_state=42, **params)
    if model_name == 'Ridge':
        return Ridge(random_state=42, **params)
    if model_name == 'ElasticNet':
        return ElasticNet(random_state=42, max_iter=5000, **params)
    if model_name == 'KernelRidge_rbf':
        return KernelRidge(kernel='rbf', **params)
    raise ValueError(model_name)


def suggest_params(trial, model_name):
    if model_name == 'ExtraTrees':
        return dict(n_estimators=trial.suggest_int('n_estimators', 100, 800),
                    max_depth=trial.suggest_int('max_depth', 3, 30),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    max_features=trial.suggest_float('max_features', 0.3, 1.0))
    if model_name == 'RandomForest':
        return dict(n_estimators=trial.suggest_int('n_estimators', 100, 800),
                    max_depth=trial.suggest_int('max_depth', 3, 30),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    max_features=trial.suggest_float('max_features', 0.3, 1.0))
    if model_name == 'HistGradientBoosting':
        return dict(max_iter=trial.suggest_int('max_iter', 50, 500),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    max_depth=trial.suggest_int('max_depth', 2, 12),
                    l2_regularization=trial.suggest_float('l2_regularization', 1e-4, 10, log=True))
    if model_name == 'LightGBM':
        return dict(n_estimators=trial.suggest_int('n_estimators', 50, 600),
                    learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                    num_leaves=trial.suggest_int('num_leaves', 7, 128),
                    min_child_samples=trial.suggest_int('min_child_samples', 2, 30),
                    reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
                    reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
                    subsample=trial.suggest_float('subsample', 0.5, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0))
    if model_name == 'SVR_rbf':
        return dict(C=trial.suggest_float('C', 1e-2, 1e3, log=True),
                    gamma=trial.suggest_float('gamma', 1e-4, 1, log=True),
                    epsilon=trial.suggest_float('epsilon', 1e-3, 1, log=True))
    if model_name == 'GradientBoosting':
        return dict(n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    max_depth=trial.suggest_int('max_depth', 2, 8),
                    subsample=trial.suggest_float('subsample', 0.5, 1.0))
    if model_name == 'Ridge':
        return dict(alpha=trial.suggest_float('alpha', 1e-3, 100, log=True))
    if model_name == 'ElasticNet':
        return dict(alpha=trial.suggest_float('alpha', 1e-4, 10, log=True),
                    l1_ratio=trial.suggest_float('l1_ratio', 0.01, 0.99))
    if model_name == 'KernelRidge_rbf':
        return dict(alpha=trial.suggest_float('alpha', 1e-4, 10, log=True),
                    gamma=trial.suggest_float('gamma', 1e-4, 1, log=True))
    raise ValueError(model_name)


def tune_model_nested(model_name, X_outer_train, y_outer_train, numeric_cols, categorical_cols,
                       n_trials=50, inner_splits=4, seed=0, timeout=None):
    """Inner-CV tuning restricted to X_outer_train/y_outer_train only. Returns
    the best params (dict) and the fitted preprocessor+model ready to predict
    on whatever outer test/val fold the caller has, WITHOUT ever having seen it.
    """
    inner_kf = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    def objective(trial):
        params = suggest_params(trial, model_name)
        scores = []
        for tr_idx, va_idx in inner_kf.split(X_outer_train):
            Xtr_raw = X_outer_train.iloc[tr_idx]
            Xva_raw = X_outer_train.iloc[va_idx]
            ytr = y_outer_train.iloc[tr_idx]
            yva = y_outer_train.iloc[va_idx]
            pre = build_preprocessor(numeric_cols, categorical_cols)
            Xtr = pre.fit_transform(Xtr_raw)
            Xva = pre.transform(Xva_raw)
            model = make_model_from_params(model_name, params)
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            scores.append(r2_score(yva, pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    # refit preprocessor + model on the FULL outer-train fold with best params
    pre_final = build_preprocessor(numeric_cols, categorical_cols)
    Xtr_final = pre_final.fit_transform(X_outer_train)
    model_final = make_model_from_params(model_name, best_params)
    model_final.fit(Xtr_final, y_outer_train)
    return best_params, study.best_value, pre_final, model_final


TUNABLE_MODELS = ['ExtraTrees', 'RandomForest', 'HistGradientBoosting', 'LightGBM',
                   'SVR_rbf', 'GradientBoosting', 'Ridge', 'ElasticNet', 'KernelRidge_rbf']


def tuned_repeated_splits(model_name, X, y, numeric_cols, categorical_cols, seeds,
                           n_trials, inner_splits=4, test_size=0.2):
    """For each outer seed: tune on the 80% train fold only (inner CV), then
    score once on the untouched 20% test fold. This is the honest 'tuned'
    version of evaluate_model_repeated_splits.
    """
    records = []
    best_params_log = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        best_params, inner_r2, pre, model = tune_model_nested(
            model_name, X_train, y_train, numeric_cols, categorical_cols,
            n_trials=n_trials, inner_splits=inner_splits, seed=seed)
        Xte = pre.transform(X_test)
        pred = model.predict(Xte)
        records.append({'seed': seed, 'r2': r2_score(y_test, pred), 'rmse': rmse(y_test, pred),
                         'mae': mean_absolute_error(y_test, pred), 'inner_cv_r2': inner_r2})
        best_params_log.append({'seed': seed, 'params': best_params})
    return pd.DataFrame(records), best_params_log


if __name__ == '__main__':
    # smoke test on water permeability with ExtraTrees, tiny budget
    import full_pipeline_part1_data as p1
    df1 = p1.canonicalize_categoricals(p1.load_sheet1())
    df1 = p1.add_chemical_family_features(df1)
    df1 = p1.add_interaction_features(df1)

    target = 'water permeability (LMH/bar)'
    all_targets = p1.PERFORMANCE_TARGETS + [p1.IRREVERSIBLE_FOULING_COL]
    feature_cols = [c for c in df1.columns if c not in all_targets + ['DOI', 'SMILES']]
    X = df1[feature_cols]
    y = df1[target]
    numeric_cols = X.select_dtypes(include='number').columns.tolist()
    categorical_cols = X.select_dtypes(exclude='number').columns.tolist()

    t0 = time.time()
    res, params_log = tuned_repeated_splits('ExtraTrees', X, y, numeric_cols, categorical_cols,
                                             seeds=[1000, 1001], n_trials=5, inner_splits=3)
    print(res)
    print("params:", params_log)
    print("smoke test time:", time.time() - t0, "s")

def tuned_repeated_splits_logtarget(model_name, X, y, numeric_cols, categorical_cols, seeds,
                                     n_trials, inner_splits=4, test_size=0.2):
    """Same as tuned_repeated_splits but trains on log1p(y), scores on original scale."""
    import numpy as np
    y_log = np.log1p(y)
    records = []
    for seed in seeds:
        X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=test_size, random_state=seed)
        y_test_orig = np.expm1(y_test_log)
        best_params, inner_r2, pre, model = tune_model_nested(
            model_name, X_train, y_train_log, numeric_cols, categorical_cols,
            n_trials=n_trials, inner_splits=inner_splits, seed=seed)
        Xte = pre.transform(X_test)
        pred_log = model.predict(Xte)
        pred_orig = np.expm1(pred_log)
        records.append({'seed': seed, 'r2': r2_score(y_test_orig, pred_orig), 'rmse': rmse(y_test_orig, pred_orig)})
    return pd.DataFrame(records)

def tune_with_train_test_metrics(model_name, X, y, numeric_cols, categorical_cols, seed=1000,
                                  n_trials=25, inner_splits=4, test_size=0.2, log_target=False):
    """Single 80/20 split (matching Gao's own protocol), real nested-CV tuning,
    returns BOTH train and test R2/RMSE on the model actually fit."""
    y_fit = np.log1p(y) if log_target else y
    X_train, X_test, y_train_fit, y_test_fit = train_test_split(X, y_fit, test_size=test_size, random_state=seed)
    best_params, inner_r2, pre, model = tune_model_nested(
        model_name, X_train, y_train_fit, numeric_cols, categorical_cols,
        n_trials=n_trials, inner_splits=inner_splits, seed=seed)
    Xtr = pre.transform(X_train)
    Xte = pre.transform(X_test)
    pred_train = model.predict(Xtr)
    pred_test = model.predict(Xte)
    if log_target:
        y_train_orig = np.expm1(y_train_fit); y_test_orig = np.expm1(y_test_fit)
        pred_train = np.expm1(pred_train); pred_test = np.expm1(pred_test)
    else:
        y_train_orig, y_test_orig = y_train_fit, y_test_fit
    return {
        'model': model_name, 'seed': seed, 'best_params': best_params,
        'train_r2': r2_score(y_train_orig, pred_train), 'train_rmse': rmse(y_train_orig, pred_train),
        'test_r2': r2_score(y_test_orig, pred_test), 'test_rmse': rmse(y_test_orig, pred_test),
        'n_train': len(y_train_orig), 'n_test': len(y_test_orig),
        'y_train': y_train_orig, 'pred_train': pred_train,
        'y_test': y_test_orig, 'pred_test': pred_test,
    }

def build_and_eval_ensemble_n(X, y, numeric_cols, categorical_cols, model_names, seed=1000, n_trials=25):
    """N-model leakage-safe weighted ensemble (weights fit on held-out slice of TRAIN only)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    Xw_tr, Xw_va, yw_tr, yw_va = train_test_split(X_train, y_train, test_size=0.25, random_state=seed)
    weights = {}; test_preds = {}; solo_r2 = {}
    for mname in model_names:
        best_params, inner_r2, pre, model = tune_model_nested(mname, X_train, y_train, numeric_cols, categorical_cols, n_trials=n_trials, seed=seed)
        pre_w = build_preprocessor(numeric_cols, categorical_cols)
        Xw_tr_t = pre_w.fit_transform(Xw_tr); Xw_va_t = pre_w.transform(Xw_va)
        model_w = make_model_from_params(mname, best_params)
        model_w.fit(Xw_tr_t, yw_tr)
        val_rmse = rmse(yw_va, model_w.predict(Xw_va_t))
        weights[mname] = 1.0/max(val_rmse,1e-6)
        Xte = pre.transform(X_test)
        test_preds[mname] = model.predict(Xte)
        solo_r2[mname] = r2_score(y_test, test_preds[mname])
    total_w = sum(weights.values())
    ens_pred = sum(weights[m]/total_w*test_preds[m] for m in model_names)
    return r2_score(y_test, ens_pred), rmse(y_test, ens_pred), weights, solo_r2

def build_and_eval_ensemble_n_with_train(X, y, numeric_cols, categorical_cols, model_names, seed=1000, n_trials=25):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    Xw_tr, Xw_va, yw_tr, yw_va = train_test_split(X_train, y_train, test_size=0.25, random_state=seed)
    weights = {}; test_preds = {}; train_preds = {}
    for mname in model_names:
        best_params, inner_r2, pre, model = tune_model_nested(mname, X_train, y_train, numeric_cols, categorical_cols, n_trials=n_trials, seed=seed)
        pre_w = build_preprocessor(numeric_cols, categorical_cols)
        Xw_tr_t = pre_w.fit_transform(Xw_tr); Xw_va_t = pre_w.transform(Xw_va)
        model_w = make_model_from_params(mname, best_params)
        model_w.fit(Xw_tr_t, yw_tr)
        val_rmse = rmse(yw_va, model_w.predict(Xw_va_t))
        weights[mname] = 1.0/max(val_rmse,1e-6)
        Xtr_full = pre.transform(X_train); Xte = pre.transform(X_test)
        train_preds[mname] = model.predict(Xtr_full)
        test_preds[mname] = model.predict(Xte)
    total_w = sum(weights.values())
    ens_train_pred = sum(weights[m]/total_w*train_preds[m] for m in model_names)
    ens_test_pred = sum(weights[m]/total_w*test_preds[m] for m in model_names)
    return {'train_r2': r2_score(y_train, ens_train_pred), 'train_rmse': rmse(y_train, ens_train_pred),
            'test_r2': r2_score(y_test, ens_test_pred), 'test_rmse': rmse(y_test, ens_test_pred), 'weights': weights}
