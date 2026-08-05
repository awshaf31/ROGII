"""Feature engineering for TVT prediction.

Every prediction is anchored to the well's own Prediction-Start (PS) point --
the last row where TVT_input is known. The models never predict absolute TVT;
they predict dTVT, the offset from that anchor, which lets one model generalise
across wells at very different absolute depths.

This module is the single source of truth for the feature pipeline, shared by
the notebooks, train_model.py, evaluate_models.py, generate_submission.py and
the Streamlit app, so inference-time features cannot drift from training-time
features.
"""

import numpy as np
import pandas as pd

from data_loader import get_all_well_ids, load_well_data

# Columns carried out of the pipeline: model inputs plus the context needed to
# reconstruct absolute TVT (ps_TVT) and to group folds by well (well_id).
OUTPUT_COLUMNS = [
    "well_id",
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "ps_TVT",
    "ps_Z",
    "ps_MD",
    "ps_GR",
    "delta_Z",
    "delta_MD",
    "dist_XY",
    "inclination_proxy",
]


def get_prediction_start_index(horz_df: pd.DataFrame) -> int:
    """Return the index of the Prediction Start (PS) point: the last row where
    TVT_input is known. Everything after it is what the model must predict."""
    if "TVT_input" not in horz_df.columns:
        return 0

    valid_indices = horz_df["TVT_input"].dropna().index
    if len(valid_indices) == 0:
        return 0
    return valid_indices[-1]


def extract_landing_zone_features(horz_df: pd.DataFrame, ps_idx: int) -> dict:
    """Read the state of the well at the PS point. These anchor values are
    broadcast onto every subsequent row as the reference frame."""
    if ps_idx == 0 and pd.isna(horz_df.loc[0, "TVT_input"]):
        # No known TVT anywhere in the well; nothing to anchor to.
        return {"ps_TVT": 0, "ps_Z": 0, "ps_MD": 0, "ps_X": 0, "ps_Y": 0, "ps_GR": 0}

    ps_row = horz_df.loc[ps_idx]

    return {
        "ps_TVT": ps_row.get("TVT_input", np.nan),
        "ps_Z": ps_row.get("Z", np.nan),
        "ps_MD": ps_row.get("MD", np.nan),
        "ps_X": ps_row.get("X", np.nan),
        "ps_Y": ps_row.get("Y", np.nan),
        "ps_GR": ps_row.get("GR", np.nan),
    }


def extract_features_for_well(
    well_id: str,
    horz_df: pd.DataFrame,
    type_df: pd.DataFrame,
    is_train: bool = True,
) -> pd.DataFrame:
    """Build the tabular feature table for a single well.

    Each row after the PS point becomes one training or prediction sample.
    Returns an empty DataFrame when the well has no rows past the PS point.

    `type_df` is accepted so callers share one signature across the pipeline;
    the current five-feature set is derived entirely from the lateral, with the
    typewell used for validation rather than as a direct feature source.
    """
    ps_idx = get_prediction_start_index(horz_df)
    ps_features = extract_landing_zone_features(horz_df, ps_idx)

    predict_df = horz_df.loc[ps_idx + 1 :].copy()
    if len(predict_df) == 0:
        return pd.DataFrame()

    for key, value in ps_features.items():
        predict_df[key] = value

    # Trajectory deltas: position relative to the PS anchor rather than absolute.
    predict_df["delta_Z"] = predict_df["Z"] - predict_df["ps_Z"]
    predict_df["delta_MD"] = predict_df["MD"] - predict_df["ps_MD"]
    delta_x = predict_df["X"] - predict_df["ps_X"]
    delta_y = predict_df["Y"] - predict_df["ps_Y"]

    # Horizontal step-out from the PS point.
    predict_df["dist_XY"] = np.sqrt(delta_x**2 + delta_y**2)

    # Normalised local dip-rate: elevation change per unit distance drilled.
    predict_df["inclination_proxy"] = predict_df["delta_Z"] / (
        predict_df["delta_MD"] + 1e-6
    )

    predict_df["well_id"] = well_id

    # GR has ~43% sensor dropout. LightGBM could split on NaN natively, but
    # Ridge and the MLP cannot, so GR is mean-imputed per well here to keep all
    # three models on identical inputs and the comparison fair.
    predict_df["GR"] = predict_df["GR"].fillna(predict_df["GR"].mean())

    output_columns = list(OUTPUT_COLUMNS)
    if is_train and "TVT" in predict_df.columns:
        output_columns.append("TVT")

    return predict_df[[c for c in output_columns if c in predict_df.columns]]


def build_feature_table(
    dataset_dir,
    split: str = "train",
    well_ids: list = None,
    add_target: bool = True,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Extract features for every well in a split and concatenate them.

    Args:
        dataset_dir: Base directory of the dataset.
        split: 'train' or 'test'.
        well_ids: Optional subset of wells; defaults to every well in the split.
        add_target: Append the `target_dTVT` column (training splits only).
        progress_every: Print progress every N wells; 0 disables progress output.

    Returns:
        A single DataFrame of all post-PS rows across the requested wells.
    """
    if well_ids is None:
        well_ids = get_all_well_ids(dataset_dir, split=split)

    is_train = split == "train"
    frames = []
    for i, well_id in enumerate(well_ids, start=1):
        horz_df, type_df = load_well_data(well_id, dataset_dir, split=split)
        well_features = extract_features_for_well(
            well_id, horz_df, type_df, is_train=is_train
        )
        if len(well_features) > 0:
            frames.append(well_features)
        if progress_every and i % progress_every == 0:
            print(f"  processed {i}/{len(well_ids)} wells...")

    dataset = pd.concat(frames, ignore_index=True)
    if add_target and "TVT" in dataset.columns:
        dataset["target_dTVT"] = dataset["TVT"] - dataset["ps_TVT"]

    return dataset
