"""
================================================================================
UF MEMBRANE ML PIPELINE — full spec, NO XGBoost, NO CatBoost anywhere.
Benchmark target: Gao et al., Environ. Sci. Technol. 2023, 57, 17831-17840.

CONFIG below controls compute budget. SMOKE_TEST=True runs a tiny version
(few seeds, few epochs, few Optuna trials) to catch bugs fast. Set
SMOKE_TEST=False for the real, full-spec run (30 seeds, full Optuna budget,
full DL training) -- that run will take a long time on CPU; a GPU machine
is strongly recommended if you have one (torch will auto-use CUDA if present).
================================================================================
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------
# CONFIG -- the ONE place to dial compute up/down
# ------------------------------------------------------------------------
SMOKE_TEST = os.environ.get('UF_SMOKE_TEST', '1') == '1'  # set env UF_SMOKE_TEST=0 for full run

if SMOKE_TEST:
    N_SEEDS = 3
    N_OPTUNA_TRIALS = 5
    N_GROUPKFOLD = 3
    MAX_LOSO_FOLDS = 5
    DL_EPOCHS = 15
    DL_ENSEMBLE_SIZE = 2
else:
    N_SEEDS = 30
    N_OPTUNA_TRIALS = 50
    N_GROUPKFOLD = 5
    MAX_LOSO_FOLDS = None  # None = run all DOI groups (53 folds for sheet1)
    DL_EPOCHS = 300
    DL_ENSEMBLE_SIZE = 7

RANDOM_SEEDS = list(range(1000, 1000 + N_SEEDS))
DATA_PATH = 'data/es2c05404_si_002.xlsx'
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/figures', exist_ok=True)

print(f"[CONFIG] SMOKE_TEST={SMOKE_TEST}  N_SEEDS={N_SEEDS}  N_OPTUNA_TRIALS={N_OPTUNA_TRIALS}  DL_EPOCHS={DL_EPOCHS}")

# ------------------------------------------------------------------------
# GAO ET AL. BENCHMARKS (hard-coded from the paper text, Table 1 & 2)
# ------------------------------------------------------------------------
GAO_BENCHMARKS = {
    'water permeability (LMH/bar)':   {'r2': 0.83, 'rmse': 68.24, 'kind': 'performance'},
    'organic compound removal (%)':   {'r2': 0.84, 'rmse': 6.60,  'kind': 'performance'},
    'flux decline ratio (%)':         {'r2': 0.78, 'rmse': 9.50,  'kind': 'performance'},
    'flux recovery ratio (%)':        {'r2': 0.62, 'rmse': 7.89,  'kind': 'performance'},
    'reversible fouling ratio (%)':   {'r2': 0.73, 'rmse': 10.33, 'kind': 'performance'},
    'overall porosity (%)':           {'r2': 0.66, 'rmse': 7.61,  'kind': 'property'},
    'mean pore radius (nm)':          {'r2': 0.87, 'rmse': 7.59,  'kind': 'property'},
    'contact angle (degree)':         {'r2': 0.76, 'rmse': 5.87,  'kind': 'property'},
}

# ==========================================================================
# SECTION 1-2: DATA LOADING + AUDIT-INFORMED CLEANING
# (header rows / sheet roles verified by hand against the actual file;
#  see 01_data_audit_report.md for the full audit)
# ==========================================================================

def load_sheet1():
    """Framework A: fabrication + operational conditions -> 5 performance targets."""
    df = pd.read_excel(DATA_PATH, sheet_name='1', header=2).dropna(how='all')
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_sheet2():
    """Stage-1 of two-stage pipeline: fabrication conditions -> 3 membrane properties."""
    df = pd.read_excel(DATA_PATH, sheet_name='2', header=1).dropna(how='all')
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_sheet3():
    """Stage-2 raw inputs: membrane properties + operational conditions -> performance."""
    df = pd.read_excel(DATA_PATH, sheet_name='3', header=1).dropna(how='all')
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={'Unnamed: 13': 'DOI'})
    return df


PERFORMANCE_TARGETS = ['water permeability (LMH/bar)', 'organic compound removal (%)',
                        'flux decline ratio (%)', 'flux recovery ratio (%)',
                        'reversible fouling ratio (%)']
IRREVERSIBLE_FOULING_COL = 'irreversible fouling ratio(%)'  # dropped: PCC=-1 with flux recovery ratio
PROPERTY_TARGETS = ['overall porosity (%)', 'mean pore radius (nm)', 'contact angle (degree)']


# ==========================================================================
# SECTION 5A: CHEMICAL-FAMILY TAXONOMY FOR THE `additive` COLUMN
# Classification derived ONLY from what each material name states (GO =
# graphene oxide -> graphene family, ZIF-8 = named MOF, etc). No invented
# property values.
# ==========================================================================
ADDITIVE_FAMILY = {
    'none': 'none',
    'GO-1': 'graphene', 'GO-2': 'graphene', 'GO-3': 'graphene', 'GO-4': 'graphene',
    'GO-6': 'graphene', 'GO-7': 'graphene', 'GO-8': 'graphene', 'GO-9': 'graphene',
    'rGO': 'graphene', 'sulfonated-GO': 'graphene', 'DPPA-GO': 'graphene',
    'Cysteine-GO': 'graphene', 'GO-SiO2-NH2': 'graphene_silica_hybrid',
    'GO-UiO-66': 'graphene_mof_hybrid', 'PDA-rGO': 'graphene', 'PDAAQ-rGO': 'graphene',
    'QSiPD-rGO': 'graphene',
    'TiO2 NPs-1': 'metal_oxide', 'TiO2 NPs-2': 'metal_oxide', 'TiO2 NPs-3': 'metal_oxide',
    'TiO2 NPs-4': 'metal_oxide', 'TiO2 NPs-5': 'metal_oxide', 'HEMA-TiO2': 'metal_oxide',
    'Ag-TiO2': 'metal_oxide', 'ZnO': 'metal_oxide', 'ZnO NPs': 'metal_oxide',
    'ZnFe2O4': 'metal_oxide', 'PDA-ZnFe2O4': 'metal_oxide', 'MnO2 NPs': 'metal_oxide',
    'Fe2O3-Mn2O3 NPs': 'metal_oxide', 'TiO2-SO3H NTs': 'metal_oxide',
    'CeO2-HNTs': 'metal_oxide_clay_hybrid',
    'ZIF-8': 'mof', 'ZIF-8 (Z1)': 'mof', 'ZIF-8 (Z2)': 'mof', 'hZIF-8': 'mof',
    'UiO-66-2': 'mof', 'UiO-66-3': 'mof',
    'MWCNT': 'cnt', 'OMWCNTs': 'cnt', 'CNT': 'cnt', 'SLS-CNT': 'cnt',
    'Sulfonated-CNT': 'cnt', 'amine-MWCNTs': 'cnt', 'PAA-g-MWCNT-TiO2': 'cnt_metal_oxide_hybrid',
    'HNTs': 'nanoclay', 'PDA-g-HANTs': 'nanoclay', 'PDA-PEI-g-HANTs': 'nanoclay',
    'PDA-TA-g-HANTs': 'nanoclay', 'Hap-BN': 'nanoclay', 'HAp/AC': 'nanoclay_carbon_hybrid',
    'Hap-AC': 'nanoclay_carbon_hybrid',
    'CNC-1': 'cellulose_nanocrystal', 'CNC-2': 'cellulose_nanocrystal', 'CNC-3': 'cellulose_nanocrystal',
    'CQDs': 'carbon_dot',
    'PMO-g-SiO2': 'silica', 'PMO-SO3H-g-SiO2-NR': 'silica', 'mesoporous SiO2 NPs': 'silica',
    'PVP-g-MMT': 'polymeric_clay',
    'LiCl': 'inorganic_salt',
    'PFSA': 'polymeric_additive', 'SPDA': 'polymeric_additive',
    'SPSF-DS10': 'polymeric_additive', 'SPSF-DS30': 'polymeric_additive',
    'SPSF-DS50': 'polymeric_additive', 'PDAAQ': 'polymeric_additive',
    'CuO-gC3N4': 'carbon_nitride_metal_oxide_hybrid', 'g-C3N4-1': 'carbon_nitride',
    'g-C3N4-2': 'carbon_nitride',
}
INORGANIC_ORGANIC = {
    'none': 'none', 'graphene': 'inorganic', 'graphene_silica_hybrid': 'inorganic',
    'graphene_mof_hybrid': 'hybrid', 'metal_oxide': 'inorganic', 'metal_oxide_clay_hybrid': 'inorganic',
    'mof': 'hybrid', 'cnt': 'inorganic', 'cnt_metal_oxide_hybrid': 'inorganic',
    'nanoclay': 'inorganic', 'nanoclay_carbon_hybrid': 'inorganic',
    'cellulose_nanocrystal': 'organic', 'carbon_dot': 'inorganic', 'silica': 'inorganic',
    'polymeric_clay': 'hybrid', 'inorganic_salt': 'inorganic', 'polymeric_additive': 'organic',
    'carbon_nitride_metal_oxide_hybrid': 'hybrid', 'carbon_nitride': 'inorganic',
}
DIMENSIONALITY = {
    'none': 'none', 'graphene': '2D', 'graphene_silica_hybrid': '2D', 'graphene_mof_hybrid': '2D',
    'metal_oxide': '0D', 'metal_oxide_clay_hybrid': '1D', 'mof': '3D', 'cnt': '1D',
    'cnt_metal_oxide_hybrid': '1D', 'nanoclay': '1D', 'nanoclay_carbon_hybrid': '1D',
    'cellulose_nanocrystal': '1D', 'carbon_dot': '0D', 'silica': '0D', 'polymeric_clay': '2D',
    'inorganic_salt': '0D', 'polymeric_additive': 'polymer_chain',
    'carbon_nitride_metal_oxide_hybrid': '2D', 'carbon_nitride': '2D',
}

def add_chemical_family_features(df, additive_col='additive'):
    df = df.copy()
    df['additive_family'] = df[additive_col].map(ADDITIVE_FAMILY).fillna('unknown')
    df['additive_inorganic_organic'] = df['additive_family'].map(INORGANIC_ORGANIC).fillna('unknown')
    df['additive_dimensionality'] = df['additive_family'].map(DIMENSIONALITY).fillna('unknown')
    df['is_graphene_family'] = df['additive_family'].isin(['graphene', 'graphene_silica_hybrid', 'graphene_mof_hybrid']).astype(int)
    df['is_mof'] = df['additive_family'].isin(['mof', 'graphene_mof_hybrid']).astype(int)
    df['is_metal_oxide'] = df['additive_family'].str.contains('metal_oxide').astype(int)
    df['is_cnt'] = df['additive_family'].str.contains('cnt').astype(int)
    df['is_polymeric_additive'] = (df['additive_family'] == 'polymeric_additive').astype(int)
    df['has_additive'] = (df[additive_col] != 'none').astype(int)
    return df

def canonicalize_categoricals(df):
    df = df.copy()
    if 'organic solvent' in df.columns:
        df['organic solvent'] = df['organic solvent'].replace({'DMAC': 'DMAc'})
    if 'organic compound' in df.columns:
        df['organic compound'] = df['organic compound'].replace(
            {'Safranine O': 'Safranin O', 'Egg albumin ': 'Egg albumin'})
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
    return df


# ==========================================================================
# SECTION 5C: PHYSICALLY-MOTIVATED INTERACTION FEATURES
# ==========================================================================
def add_interaction_features(df):
    df = df.copy()
    def safe_col(name):
        return df[name] if name in df.columns else None

    pairs = [
        ('P_wt%', 'additive wt% (additive to polymer)', 'polymer_x_additive'),
        ('P_wt%', 'pore maker wt% (pore maker to total weight)', 'polymer_x_poremaker'),
        ('additive wt% (additive to polymer)', 'pore maker wt% (pore maker to total weight)', 'additive_x_poremaker'),
        ('TMP (bar)', 'P_wt%', 'tmp_x_polymer'),
        ('TMP (bar)', 'additive wt% (additive to polymer)', 'tmp_x_additive'),
        ('pore maker molecular weight (Da)', 'pore maker wt% (pore maker to total weight)', 'poremakerMW_x_poremakerwt'),
    ]
    for a, b, name in pairs:
        if a in df.columns and b in df.columns:
            df[name] = df[a] * df[b]
    return df


if __name__ == '__main__':
    df1 = canonicalize_categoricals(load_sheet1())
    df1 = add_chemical_family_features(df1)
    df1 = add_interaction_features(df1)
    print("Sheet1 shape after cleaning + feature engineering:", df1.shape)
    print("New columns added:", [c for c in df1.columns if c not in load_sheet1().columns])
    df1.to_pickle(f'{OUT_DIR}/sheet1_engineered.pkl')

    df2 = canonicalize_categoricals(load_sheet2())
    df2 = add_chemical_family_features(df2)
    print("Sheet2 shape:", df2.shape)
    df2.to_pickle(f'{OUT_DIR}/sheet2_engineered.pkl')

    df3 = canonicalize_categoricals(load_sheet3())
    print("Sheet3 shape:", df3.shape)
    df3.to_pickle(f'{OUT_DIR}/sheet3_engineered.pkl')
