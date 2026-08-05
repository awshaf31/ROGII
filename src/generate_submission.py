"""Predict TVT for every masked row in the test split and write a Kaggle
submission file.

Usage:
    python src/generate_submission.py
"""

import joblib
import pandas as pd

from config import DATA_DIR, FEATURES, MODEL_PATH, OUTPUTS_DIR
from data_loader import get_all_well_ids, load_well_data
from features import extract_features_for_well

SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"


def main():
    if not MODEL_PATH.exists():
        print(f"Error: model not found at {MODEL_PATH}. Run train_model.py first.")
        return

    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

    test_well_ids = get_all_well_ids(DATA_DIR, split="test")
    print(f"Found {len(test_well_ids)} test wells to process.")

    submission_rows = []
    for i, well_id in enumerate(test_well_ids, start=1):
        horz_df, type_df = load_well_data(well_id, DATA_DIR, split="test")
        predict_df = extract_features_for_well(
            well_id, horz_df, type_df, is_train=False
        )
        if len(predict_df) == 0:
            continue

        # dTVT is an offset; add the PS anchor back to recover absolute TVT.
        preds_dtvt = model.predict(predict_df[FEATURES])
        predicted_tvt = predict_df["ps_TVT"] + preds_dtvt

        # Kaggle row identifier format: {well_id}_{row_index}
        submission_rows.extend(
            {"id": f"{well_id}_{orig_idx}", "tvt": tvt_val}
            for orig_idx, tvt_val in zip(predict_df.index, predicted_tvt)
        )

        if i % 50 == 0:
            print(f"Processed {i} test wells...")

    sub_df = pd.DataFrame(submission_rows)
    print(f"\nSubmission complete. Total rows predicted: {len(sub_df)}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
