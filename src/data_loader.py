"""File I/O for the ROGII dataset: locating wells and pairing each horizontal
lateral with its vertical typewell."""

from pathlib import Path
from typing import Iterator, Union

import pandas as pd

PathLike = Union[str, Path]


def load_well_data(well_id: str, dataset_dir: PathLike, split: str = "train"):
    """Load the horizontal well and its paired typewell for a given well_id.

    Args:
        well_id: ID of the well (e.g. '000d7d20').
        dataset_dir: Base directory of the dataset.
        split: 'train' or 'test'.

    Returns:
        tuple: (horizontal_df, typewell_df)
    """
    base_path = Path(dataset_dir) / split

    horz_path = base_path / f"{well_id}__horizontal_well.csv"
    type_path = base_path / f"{well_id}__typewell.csv"

    if not horz_path.exists() or not type_path.exists():
        raise FileNotFoundError(f"Files for well {well_id} not found in {base_path}")

    return pd.read_csv(horz_path), pd.read_csv(type_path)


def check_data_integrity(horz_df: pd.DataFrame, type_df: pd.DataFrame) -> dict:
    """Report GR missingness, row counts, and whether the lateral's known TVT
    range sits inside the typewell's TVT range.

    That containment is the structural precondition the typewell-anchored
    feature pipeline depends on.
    """
    type_tvt_range = (type_df["TVT"].min(), type_df["TVT"].max())

    results = {
        "horz_gr_nans": horz_df["GR"].isna().sum(),
        "type_gr_nans": type_df["GR"].isna().sum(),
        "type_tvt_range": type_tvt_range,
    }

    # Absent on strict test splits, where no part of TVT is revealed.
    if "TVT_input" in horz_df.columns:
        horz_tvt_min = horz_df["TVT_input"].min()
        horz_tvt_max = horz_df["TVT_input"].max()
        results["horz_tvt_input_range"] = (horz_tvt_min, horz_tvt_max)
        results["horz_tvt_in_typewell_bounds"] = (
            horz_tvt_min >= type_tvt_range[0] and horz_tvt_max <= type_tvt_range[1]
        )

    results["horz_length"] = len(horz_df)
    results["type_length"] = len(type_df)

    return results


def get_all_well_ids(dataset_dir: PathLike, split: str = "train") -> list:
    """Return the sorted unique well IDs present in the given split."""
    base_path = Path(dataset_dir) / split
    well_ids = {
        f.name.split("__")[0]
        for f in base_path.iterdir()
        if f.name.endswith("__horizontal_well.csv")
    }
    return sorted(well_ids)


def load_all_wells_generator(dataset_dir: PathLike, split: str = "train") -> Iterator:
    """Yield (well_id, horz_df, type_df) for every well in a split."""
    for well_id in get_all_well_ids(dataset_dir, split):
        horz_df, type_df = load_well_data(well_id, dataset_dir, split)
        yield well_id, horz_df, type_df
