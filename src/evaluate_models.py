"""Apples-to-apples comparison of Ridge / LightGBM / PyTorch MLP (Section 5).

All three models are trained on the same rows and the same GroupKFold splits,
grouped by well_id so no well ever appears in both train and validation. RMSE
is measured in feet on reconstructed absolute TVT, not on the dTVT offset.

Usage:
    python src/evaluate_models.py

Outputs:
    outputs/results_full_comparison.json   per-fold and mean RMSE per model
    reports/figures/05_model_comparison.png
    models/lgbm_model_full.joblib          LightGBM retrained on all wells
"""

import os

# Must precede the torch / lightgbm imports: keeps OpenMP from over-subscribing
# threads when both libraries are loaded in the same process.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import time

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_DIR,
    FEATURES,
    FIGURES_DIR,
    LGB_PARAMS,
    MODELS_DIR,
    N_SPLITS,
    OUTPUTS_DIR,
    TARGET,
)
from features import build_feature_table

# Set to an integer (e.g. 200) for a fast smoke run; None uses all 773 wells.
WELLS_LIMIT = None

MLP_EPOCHS = 20
MLP_BATCH_SIZE = 256
MLP_LEARNING_RATE = 0.005

RESULTS_PATH = OUTPUTS_DIR / "results_full_comparison.json"
CHART_PATH = FIGURES_DIR / "05_model_comparison.png"
FULL_MODEL_PATH = MODELS_DIR / "lgbm_model_full.joblib"

# LightGBM shares the process with PyTorch here, so it is pinned to one thread.
CV_LGB_PARAMS = {**LGB_PARAMS, "n_jobs": 1}


class WellboreMLP(nn.Module):
    """Deep-learning comparator: a small MLP over the five trajectory features."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def fit_ridge(X_train, y_train, X_val):
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    return ridge.predict(X_val)


def fit_lightgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMRegressor(**CV_LGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model.predict(X_val)


def fit_mlp(X_train, y_train, X_val, device):
    """Train the MLP on standardised inputs and return validation predictions."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(
        y_train.values.reshape(-1, 1), dtype=torch.float32
    ).to(device)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=MLP_BATCH_SIZE, shuffle=True
    )

    model = WellboreMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(MLP_EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(X_val_t).cpu().numpy().flatten()


def save_comparison_chart(summary, path):
    names = ["Ridge Regression", "LightGBM", "PyTorch MLP"]
    keys = ["ridge", "lightgbm", "mlp"]
    means = [summary[k]["mean_rmse"] for k in keys]
    stds = [summary[k]["std_rmse"] for k in keys]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        names, means, yerr=stds, capsize=6, color=["#4C72B0", "#55A868", "#C44E52"]
    )
    for bar, mean in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.3,
            f"{mean:.2f}",
            ha="center",
            fontweight="bold",
        )
    plt.ylabel("Cross-Validation RMSE (feet)")
    plt.title(
        f"Model Comparison: {summary['n_wells']} wells, "
        f"{N_SPLITS}-fold GroupKFold (lower is better)"
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    well_ids = None
    if WELLS_LIMIT is not None:
        from data_loader import get_all_well_ids

        well_ids = get_all_well_ids(DATA_DIR, split="train")[:WELLS_LIMIT]

    print("Extracting features (this takes a few minutes on all 773 wells)...")
    t0 = time.time()
    dataset = build_feature_table(DATA_DIR, split="train", well_ids=well_ids)
    print(f"Dataset shape: {dataset.shape}  (took {time.time() - t0:.1f}s)")

    X = dataset[FEATURES]
    y = dataset[TARGET]
    ps_tvt = dataset["ps_TVT"]
    true_tvt = dataset["TVT"]

    # One shared split object, reused by all three models, so the comparison is fair.
    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X, y, groups=dataset["well_id"]))

    device = torch.device("cpu")
    print(f"Using device for MLP: {device}")

    results = {"ridge": [], "lightgbm": [], "mlp": []}

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        ps_val = ps_tvt.iloc[val_idx].values
        true_val = true_tvt.iloc[val_idx].values

        # Ridge and the MLP cannot accept NaN; LightGBM splits on it natively.
        X_train_filled = X_train.fillna(0)
        X_val_filled = X_val.fillna(0)

        ridge_preds = fit_ridge(X_train_filled, y_train, X_val_filled)
        results["ridge"].append(root_mean_squared_error(true_val, ps_val + ridge_preds))
        print(f"  Ridge RMSE:    {results['ridge'][-1]:.2f} ft")

        lgbm_preds = fit_lightgbm(X_train, y_train, X_val, y_val)
        results["lightgbm"].append(
            root_mean_squared_error(true_val, ps_val + lgbm_preds)
        )
        print(f"  LightGBM RMSE: {results['lightgbm'][-1]:.2f} ft")

        mlp_preds = fit_mlp(X_train_filled, y_train, X_val_filled, device)
        results["mlp"].append(root_mean_squared_error(true_val, ps_val + mlp_preds))
        print(f"  MLP RMSE:      {results['mlp'][-1]:.2f} ft")

    summary = {}
    print("\n===== FINAL SUMMARY (same wells, same folds, all 3 models) =====")
    for name, scores in results.items():
        summary[name] = {
            "fold_scores": scores,
            "mean_rmse": float(np.mean(scores)),
            "std_rmse": float(np.std(scores)),
        }
        folds = [f"{s:.2f}" for s in scores]
        print(
            f"{name:>10}: {summary[name]['mean_rmse']:.2f} "
            f"+/- {summary[name]['std_rmse']:.2f} ft   (folds: {folds})"
        )

    summary["n_wells"] = len(dataset["well_id"].unique())
    summary["n_rows"] = int(len(dataset))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {RESULTS_PATH}")

    save_comparison_chart(summary, CHART_PATH)
    print(f"Saved {CHART_PATH}")

    print("\nRetraining final LightGBM on the full dataset for deployment...")
    final_model = lgb.LGBMRegressor(**CV_LGB_PARAMS)
    final_model.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, FULL_MODEL_PATH)
    print(f"Saved {FULL_MODEL_PATH}")


if __name__ == "__main__":
    main()
