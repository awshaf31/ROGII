"""Per-well error analysis for the report (Section 5.3).

Scores the production model independently on each training well, then tests
whether prediction error correlates with post-PS dip variability or with mean
horizontal offset from the PS anchor.

Usage:
    python src/error_analysis.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from config import DATA_DIR, FEATURES, FULL_MODEL_PATH, OUTPUTS_DIR
from data_loader import get_all_well_ids, load_well_data
from features import extract_features_for_well

RESULTS_PATH = OUTPUTS_DIR / "per_well_error_analysis.csv"


def main():
    model = joblib.load(FULL_MODEL_PATH)
    well_ids = get_all_well_ids(DATA_DIR, split="train")

    per_well_results = []
    for well_id in well_ids:
        horz_df, type_df = load_well_data(well_id, DATA_DIR, split="train")
        df = extract_features_for_well(well_id, horz_df, type_df, is_train=True)
        if len(df) == 0:
            continue

        predicted_tvt = df["ps_TVT"] + model.predict(df[FEATURES])
        per_well_results.append(
            {
                "well_id": well_id,
                "rmse": np.sqrt(mean_squared_error(df["TVT"], predicted_tvt)),
                "dip_variability": df["inclination_proxy"].std(),
                "mean_offset_ft": df["dist_XY"].mean(),
            }
        )

    results_df = pd.DataFrame(per_well_results)

    print(f"Wells scored: {len(results_df)}")
    print(
        f"Mean per-well RMSE: {results_df['rmse'].mean():.2f} ft "
        f"(std {results_df['rmse'].std():.2f}, "
        f"range {results_df['rmse'].min():.2f}-{results_df['rmse'].max():.2f})"
    )
    print(
        "Correlation(RMSE, dip variability): "
        f"{results_df['rmse'].corr(results_df['dip_variability']):.3f}"
    )
    print(
        "Correlation(RMSE, mean typewell offset): "
        f"{results_df['rmse'].corr(results_df['mean_offset_ft']):.3f}"
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved per-well results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
