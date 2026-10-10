"""
DROP-IN REPLACEMENT for `_cat_encoded_column_transformer` in modeling.py.

Change: only `additive` (72 raw levels, the genuinely high-cardinality column)
gets LeaveOneOutEncoder. Everything else (Polymer, pore maker, organic solvent,
organic compound, foulant, additive_material_class, additive_dimensionality --
all <= ~20 levels) gets one-hot instead. Verified on this dataset: one-hot
beats uniform LOO-encoding by ~0.05 R2 on ExtraTrees, same model/CV/data
(see chat for the controlled test). Still 100% leakage-safe: both encoders
sit inside sklearn Pipeline/ColumnTransformer objects that get cloned and
refit on every CV training fold by the harness in evaluation.py -- nothing
about that changes here.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import LeaveOneOutEncoder

# The only column(s) worth target-encoding. Add a column here only if it has
# double-digit-plus cardinality; everything else should be one-hot.
HIGH_CARDINALITY_COLS = ["additive"]


def _cat_encoded_column_transformer(categorical_cols, numeric_cols, impute=False, scale=False):
    high_card = [c for c in categorical_cols if c in HIGH_CARDINALITY_COLS]
    low_card = [c for c in categorical_cols if c not in HIGH_CARDINALITY_COLS]

    num_steps = []
    if impute:
        num_steps.append(("impute", SimpleImputer(strategy="median")))
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps) if num_steps else "passthrough"

    transformers = []
    if high_card:
        transformers.append(("cat_loo", Pipeline([("loo", LeaveOneOutEncoder(cols=None, sigma=0.05))]), high_card))
    if low_card:
        transformers.append(("cat_onehot", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), low_card))
    transformers.append(("num", num_pipe, numeric_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")
