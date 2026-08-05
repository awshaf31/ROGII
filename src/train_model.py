"""Train the production LightGBM model on all training wells and save it.

Usage:
    python src/train_model.py
"""

import joblib
import lightgbm as lgb

from config import DATA_DIR, FEATURES, LGB_PARAMS, MODEL_PATH, MODELS_DIR, TARGET
from features import build_feature_table


def main():
    print("Extracting features for all training wells. This may take a moment...")
    dataset = build_feature_table(DATA_DIR, split="train")
    print(f"Final dataset shape: {dataset.shape}")

    X = dataset[FEATURES]
    y = dataset[TARGET]

    print("Training LightGBM model on the entire dataset...")
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")


if __name__ == "__main__":
    main()
