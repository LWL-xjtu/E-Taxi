from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DataConfig
from .utils import dump_json, ensure_dir, load_json

PICKUP_TIME_CANDIDATES = ["tpep_pickup_datetime", "pickup_datetime"]
DROPOFF_TIME_CANDIDATES = ["tpep_dropoff_datetime", "dropoff_datetime"]
FARE_CANDIDATES = ["total_amount", "fare_amount"]
COORDINATE_COLUMNS = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
LOCATION_ID_COLUMNS = ["PULocationID", "DOLocationID"]


@dataclass(slots=True)
class PreparedDataset:
    root: Path
    metadata: dict[str, Any]
    splits: dict[str, pd.DataFrame]

    @classmethod
    def load(cls, root: str | Path) -> "PreparedDataset":
        dataset_root = Path(root)
        metadata = load_json(dataset_root / "metadata.json")
        splits = {
            split: pd.read_parquet(dataset_root / f"{split}.parquet")
            for split in ("train", "val", "test")
        }
        for frame in splits.values():
            frame["time_bin"] = pd.to_datetime(frame["time_bin"], utc=False)
        return cls(root=dataset_root, metadata=metadata, splits=splits)


def _pick_first_existing(columns: list[str], frame: pd.DataFrame) -> str:
    for column in columns:
        if column in frame.columns:
            return column
    joined = ", ".join(columns)
    raise ValueError(f"Missing required columns. Expected one of: {joined}")


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    pickup_time = _pick_first_existing(PICKUP_TIME_CANDIDATES, frame)
    dropoff_time = _pick_first_existing(DROPOFF_TIME_CANDIDATES, frame)
    fare = _pick_first_existing(FARE_CANDIDATES, frame)

    canonical = pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(frame[pickup_time]),
            "dropoff_datetime": pd.to_datetime(frame[dropoff_time]),
            "mean_fare": pd.to_numeric(frame[fare], errors="coerce"),
        }
    )

    has_coordinates = all(column in frame.columns for column in COORDINATE_COLUMNS)
    has_location_ids = all(column in frame.columns for column in LOCATION_ID_COLUMNS)

    if has_coordinates:
        for column in COORDINATE_COLUMNS:
            canonical[column] = pd.to_numeric(frame[column], errors="coerce")
        canonical["mapping_mode"] = "coordinates"
    elif has_location_ids:
        canonical["pickup_location_id"] = pd.to_numeric(
            frame["PULocationID"], errors="coerce"
        ).astype("Int64")
        canonical["dropoff_location_id"] = pd.to_numeric(
            frame["DOLocationID"], errors="coerce"
        ).astype("Int64")
        canonical["mapping_mode"] = "location_id_projection"
    else:
        raise ValueError(
            "The input parquet must contain either coordinate columns or PULocationID/"
            "DOLocationID columns."
        )

    canonical["trip_minutes"] = (
        canonical["dropoff_datetime"] - canonical["pickup_datetime"]
    ).dt.total_seconds() / 60.0
    canonical = canonical.dropna(subset=["pickup_datetime", "dropoff_datetime", "mean_fare"])
    canonical = canonical[canonical["trip_minutes"] > 0]
    canonical = canonical[canonical["mean_fare"] >= 0]
    return canonical.reset_index(drop=True)


def _grid_from_coordinates(frame: pd.DataFrame, config: DataConfig) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    lon_series = pd.concat([frame["pickup_longitude"], frame["dropoff_longitude"]]).dropna()
    lat_series = pd.concat([frame["pickup_latitude"], frame["dropoff_latitude"]]).dropna()
    if lon_series.empty or lat_series.empty:
        raise ValueError("Coordinate mapping requested but no usable latitude/longitude values were found.")

    min_lon, max_lon = lon_series.min(), lon_series.max()
    min_lat, max_lat = lat_series.min(), lat_series.max()
    lon_den = max(max_lon - min_lon, 1e-6)
    lat_den = max(max_lat - min_lat, 1e-6)

    def encode(longitudes: pd.Series, latitudes: pd.Series) -> pd.Series:
        col = np.floor((longitudes - min_lon) / lon_den * config.grid_cols).clip(0, config.grid_cols - 1)
        row = np.floor((latitudes - min_lat) / lat_den * config.grid_rows).clip(0, config.grid_rows - 1)
        return (row.astype(int) * config.grid_cols + col.astype(int)).astype(int)

    pickup_cells = encode(frame["pickup_longitude"], frame["pickup_latitude"])
    dropoff_cells = encode(frame["dropoff_longitude"], frame["dropoff_latitude"])
    metadata = {
        "mapping_mode": "coordinates",
        "coordinate_bounds": {
            "min_lon": float(min_lon),
            "max_lon": float(max_lon),
            "min_lat": float(min_lat),
            "max_lat": float(max_lat),
        },
    }
    return pickup_cells, dropoff_cells, metadata


def _grid_from_location_ids(
    frame: pd.DataFrame, config: DataConfig
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    location_ids = pd.concat(
        [frame["pickup_location_id"], frame["dropoff_location_id"]]
    ).dropna()
    unique_ids = sorted(int(value) for value in location_ids.unique())
    if not unique_ids:
        raise ValueError("No usable PULocationID/DOLocationID values were found.")

    mapping: dict[int, int] = {}
    total = len(unique_ids)
    for rank, location_id in enumerate(unique_ids):
        cell = int(np.floor(rank / max(total, 1) * config.cell_count))
        mapping[location_id] = min(cell, config.cell_count - 1)

    pickup_cells = frame["pickup_location_id"].fillna(unique_ids[0]).astype(int).map(mapping)
    dropoff_cells = frame["dropoff_location_id"].fillna(unique_ids[0]).astype(int).map(mapping)
    metadata = {
        "mapping_mode": "location_id_projection",
        "location_id_to_cell": {str(key): value for key, value in mapping.items()},
    }
    return pickup_cells.astype(int), dropoff_cells.astype(int), metadata


def _assign_grid_cells(frame: pd.DataFrame, config: DataConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame["mapping_mode"].iloc[0] == "coordinates":
        pickup_cells, dropoff_cells, metadata = _grid_from_coordinates(frame, config)
    else:
        pickup_cells, dropoff_cells, metadata = _grid_from_location_ids(frame, config)

    enriched = frame.copy()
    enriched["pickup_cell"] = pickup_cells
    enriched["dropoff_cell"] = dropoff_cells
    return enriched, metadata


def _select_small_sample(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    working = frame.copy()
    working["service_date"] = working["pickup_datetime"].dt.date.astype(str)
    unique_days = sorted(working["service_date"].unique())
    selected_days = unique_days[: config.use_first_n_days]
    working = working[working["service_date"].isin(selected_days)].copy()
    working["hour"] = working["pickup_datetime"].dt.hour
    working = working[
        (working["hour"] >= config.start_hour) & (working["hour"] < config.end_hour)
    ]
    working["time_bin"] = working["pickup_datetime"].dt.floor(f"{config.step_minutes}min")
    return working.reset_index(drop=True)


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(
            ["service_date", "time_bin", "pickup_cell", "dropoff_cell"], as_index=False
        )
        .agg(
            demand_count=("pickup_cell", "size"),
            mean_trip_minutes=("trip_minutes", "mean"),
            mean_fare=("mean_fare", "mean"),
        )
        .sort_values(["service_date", "time_bin", "pickup_cell", "dropoff_cell"])
        .reset_index(drop=True)
    )
    return grouped


def _build_metadata(
    processed: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    config: DataConfig,
    mapping_metadata: dict[str, Any],
    charge_station_count: int,
) -> dict[str, Any]:
    train_frame = split_frames["train"]
    demand_totals = (
        train_frame.groupby("pickup_cell")["demand_count"].sum().reindex(
            range(config.cell_count), fill_value=0
        )
    )
    demand_priors = (
        demand_totals / max(float(demand_totals.sum()), 1.0)
    ).round(8)
    charge_stations = (
        demand_totals.sort_values(ascending=False)
        .head(min(charge_station_count, config.cell_count))
        .index.astype(int)
        .tolist()
    )
    split_dates = {
        split: sorted(frame["service_date"].unique().tolist()) for split, frame in split_frames.items()
    }
    return {
        "data": {
            "grid_rows": config.grid_rows,
            "grid_cols": config.grid_cols,
            "cell_count": config.cell_count,
            "step_minutes": config.step_minutes,
            "start_hour": config.start_hour,
            "end_hour": config.end_hour,
            "episode_steps": config.episode_steps,
        },
        "splits": split_dates,
        "charge_stations": charge_stations,
        "zone_demand_priors": demand_priors.tolist(),
        "global_defaults": {
            "mean_trip_minutes": float(processed["trip_minutes"].mean()),
            "mean_fare": float(processed["mean_fare"].mean()),
        },
        "mapping": mapping_metadata,
    }


def prepare_nyc_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    config: DataConfig,
    charge_station_count: int = 5,
) -> PreparedDataset:
    source = pd.read_parquet(input_path)
    canonical = _canonicalize_columns(source)
    canonical = _select_small_sample(canonical, config)
    processed, mapping_metadata = _assign_grid_cells(canonical, config)

    selected_days = sorted(processed["service_date"].unique())
    expected_days = config.train_days + config.val_days + config.test_days
    if len(selected_days) < expected_days:
        raise ValueError(
            f"Need at least {expected_days} distinct days after filtering, found {len(selected_days)}."
        )

    split_dates = {
        "train": selected_days[: config.train_days],
        "val": selected_days[config.train_days : config.train_days + config.val_days],
        "test": selected_days[
            config.train_days + config.val_days : config.train_days + config.val_days + config.test_days
        ],
    }

    aggregated = _aggregate(processed)
    split_frames = {
        split: aggregated[aggregated["service_date"].isin(days)].copy().reset_index(drop=True)
        for split, days in split_dates.items()
    }

    dataset_root = ensure_dir(output_dir)
    for split, frame in split_frames.items():
        frame.to_parquet(dataset_root / f"{split}.parquet", index=False)

    metadata = _build_metadata(
        processed,
        split_frames,
        config,
        mapping_metadata,
        charge_station_count=charge_station_count,
    )
    dump_json(metadata, dataset_root / "metadata.json")
    return PreparedDataset(root=dataset_root, metadata=metadata, splits=split_frames)
