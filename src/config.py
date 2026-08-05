"""Shared paths and model constants for the ROGII TVT prediction pipeline.

Every path is resolved from this file's location, so the scripts and notebooks
run correctly regardless of the working directory they are launched from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Trained on all 773 wells by train_model.py; served by the Streamlit app.
MODEL_PATH = MODELS_DIR / "lgbm_model.joblib"
# Trained on all 773 wells by evaluate_models.py, after cross-validation.
FULL_MODEL_PATH = MODELS_DIR / "lgbm_model_full.joblib"

# The five PS-anchored trajectory features every model is trained on.
FEATURES = ["delta_Z", "delta_MD", "dist_XY", "GR", "inclination_proxy"]

# Predicted target: the TVT offset from the well's Prediction-Start anchor.
TARGET = "target_dTVT"

RANDOM_STATE = 42
N_SPLITS = 5

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "n_estimators": 300,
    "num_leaves": 31,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
