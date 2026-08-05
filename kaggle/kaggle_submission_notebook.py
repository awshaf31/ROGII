"""Self-contained Kaggle notebook script (Section 5.5).

This is the exact code pasted into the Kaggle Notebook that produced the
15.077 ft public leaderboard score. It deliberately duplicates the logic of
src/data_loader.py and src/features.py rather than importing them, because a
Kaggle code competition runs in an isolated environment with no access to this
repository.

Divergence to be aware of: this version does NOT mean-impute missing GR, so it
relies on LightGBM's native NaN handling at tree-split time, whereas
src/features.py imputes GR to keep Ridge and the MLP comparable. Do not
"fix" this without re-running the Kaggle submission -- the reported score
was produced by the code below as it stands.

Usage:
    Paste into a Kaggle Notebook, or run locally from the project root.
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

# ==========================================
# 1. DATA LOADER FUNCTIONS
# ==========================================
def get_all_well_ids(dataset_dir: str, split: str = 'train') -> list:
    base_path = os.path.join(dataset_dir, split)
    files = os.listdir(base_path)
    well_ids = [f.split('__')[0] for f in files if f.endswith('__horizontal_well.csv')]
    return sorted(list(set(well_ids)))

def load_well_data(well_id: str, dataset_dir: str, split: str = 'train'):
    base_path = os.path.join(dataset_dir, split)
    horz_path = os.path.join(base_path, f"{well_id}__horizontal_well.csv")
    type_path = os.path.join(base_path, f"{well_id}__typewell.csv")
    horz_df = pd.read_csv(horz_path)
    type_df = pd.read_csv(type_path)
    return horz_df, type_df

# ==========================================
# 2. FEATURE ENGINEERING FUNCTIONS
# ==========================================
def get_prediction_start_index(horz_df: pd.DataFrame) -> int:
    if 'TVT_input' not in horz_df.columns: return 0
    valid_indices = horz_df['TVT_input'].dropna().index
    return valid_indices[-1] if len(valid_indices) > 0 else 0

def extract_landing_zone_features(horz_df: pd.DataFrame, ps_idx: int) -> dict:
    if ps_idx == 0 and pd.isna(horz_df.loc[0, 'TVT_input']):
        return {'ps_TVT': 0, 'ps_Z': 0, 'ps_MD': 0, 'ps_X': 0, 'ps_Y': 0, 'ps_GR': 0}
    ps_row = horz_df.loc[ps_idx]
    return {
        'ps_TVT': ps_row.get('TVT_input', np.nan),
        'ps_Z': ps_row.get('Z', np.nan),
        'ps_MD': ps_row.get('MD', np.nan),
        'ps_X': ps_row.get('X', np.nan),
        'ps_Y': ps_row.get('Y', np.nan),
        'ps_GR': ps_row.get('GR', np.nan)
    }

def extract_features_for_well(well_id: str, horz_df: pd.DataFrame, type_df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    ps_idx = get_prediction_start_index(horz_df)
    ps_features = extract_landing_zone_features(horz_df, ps_idx)
    predict_df = horz_df.loc[ps_idx + 1:].copy()
    if len(predict_df) == 0: return pd.DataFrame()
    
    for k, v in ps_features.items(): predict_df[k] = v
        
    predict_df['delta_Z'] = predict_df['Z'] - predict_df['ps_Z']
    predict_df['delta_MD'] = predict_df['MD'] - predict_df['ps_MD']
    predict_df['delta_X'] = predict_df['X'] - predict_df['ps_X']
    predict_df['delta_Y'] = predict_df['Y'] - predict_df['ps_Y']
    predict_df['dist_XY'] = np.sqrt(predict_df['delta_X']**2 + predict_df['delta_Y']**2)
    predict_df['inclination_proxy'] = predict_df['delta_Z'] / (predict_df['delta_MD'] + 1e-6)
    
    feature_cols = ['MD', 'X', 'Y', 'Z', 'GR', 'ps_TVT', 'ps_Z', 'ps_MD', 'ps_GR', 'delta_Z', 'delta_MD', 'dist_XY', 'inclination_proxy']
    if is_train and 'TVT' in predict_df.columns: feature_cols.append('TVT')
    
    exist_cols = [c for c in feature_cols if c in predict_df.columns]
    return predict_df[exist_cols]

# ==========================================
# 3. TRAINING & INFERENCE PIPELINE
# ==========================================

print("--- STARTING KAGGLE PIPELINE ---")

# Auto-discover the correct dataset path in Kaggle
DATA_DIR = None
if os.path.exists('/kaggle/input'):
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'train' in dirs and 'test' in dirs:
            DATA_DIR = root
            break

if DATA_DIR is None:
    # Fallback if running locally from the project root
    DATA_DIR = 'data'
    if not os.path.exists(DATA_DIR):
        print("ERROR: Could not find the dataset. If on Kaggle, make sure you clicked 'Add Data' and added the competition dataset!")
        raise FileNotFoundError("Dataset not found.")

print(f"Using Dataset Path: {DATA_DIR}")

train_well_ids = get_all_well_ids(DATA_DIR, split='train')
print(f"Extracting features for {len(train_well_ids)} training wells...")

all_features = []
for wid in train_well_ids:
    h_df, t_df = load_well_data(wid, DATA_DIR, split='train')
    df_feats = extract_features_for_well(wid, h_df, t_df, is_train=True)
    if len(df_feats) > 0: all_features.append(df_feats)

dataset = pd.concat(all_features, ignore_index=True)
dataset['target_dTVT'] = dataset['TVT'] - dataset['ps_TVT']

model_features = ['delta_Z', 'delta_MD', 'dist_XY', 'GR', 'inclination_proxy']
X_train = dataset[model_features]
y_train = dataset['target_dTVT']

print("Training LightGBM Ensemble...")
model = lgb.LGBMRegressor(objective='regression', n_estimators=300, num_leaves=31, learning_rate=0.05, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Generating predictions for hidden test set...")
test_well_ids = get_all_well_ids(DATA_DIR, split='test')
submission_rows = []

for wid in test_well_ids:
    h_df, t_df = load_well_data(wid, DATA_DIR, split='test')
    predict_df = extract_features_for_well(wid, h_df, t_df, is_train=False)
    if len(predict_df) == 0: continue
    
    X_test = predict_df[model_features].copy()
    preds_dTVT = model.predict(X_test)
    predicted_tvt = predict_df['ps_TVT'] + preds_dTVT
    
    for orig_idx, tvt_val in zip(predict_df.index, predicted_tvt):
        submission_rows.append({'id': f"{wid}_{orig_idx}", 'tvt': tvt_val})

sub_df = pd.DataFrame(submission_rows)
# Save to the Kaggle working directory
sub_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Saved submission.csv successfully! You are ready to submit.")
