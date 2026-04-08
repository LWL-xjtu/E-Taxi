from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DataConfig
from .utils import dump_json, ensure_dir, load_json

PICKUP_TIME_CANDIDATES = [
    "tpep_pickup_datetime",
    "pickup_datetime",
    "lpep_pickup_datetime",
]
DROPOFF_TIME_CANDIDATES = [
    "tpep_dropoff_datetime",
    "dropoff_datetime",
    "lpep_dropoff_datetime",
]
FARE_CANDIDATES = ["fare_amount", "fare"]
REVENUE_CANDIDATES = ["total_amount", "fare_amount", "fare"]
DISTANCE_CANDIDATES = ["trip_distance", "distance"]
COORDINATE_COLUMNS = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
LOCATION_ID_COLUMNS = ["PULocationID", "DOLocationID"]
MILES_TO_KM = 1.60934
MAX_TLC_LOCATION_ID = 263
ACTIONS_PER_ZONE = 4


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
            if "time_bin" in frame.columns:
                frame["time_bin"] = pd.to_datetime(frame["time_bin"], utc=False)
        return cls(root=dataset_root, metadata=metadata, splits=splits)

    def get_day_orders(self, date: str | pd.Timestamp, split: str = "train") -> pd.DataFrame:
        target_date = pd.Timestamp(date).date().isoformat()
        frame = self.splits[split]
        return frame[frame["service_date"] == target_date].copy().reset_index(drop=True)

    def get_orders_for_day_and_bin(
        self,
        date: str | pd.Timestamp,
        time_bin: int | str | pd.Timestamp,
        split: str = "train",
    ) -> pd.DataFrame:
        day_orders = self.get_day_orders(date=date, split=split)
        if isinstance(time_bin, (int, np.integer)):
            bin_index = int(time_bin)
        else:
            timestamp = pd.Timestamp(time_bin)
            bin_index = int(timestamp.hour * 60 // self.metadata["data"]["step_minutes"] + timestamp.minute // self.metadata["data"]["step_minutes"])
        return day_orders[day_orders["time_bin_index"] == bin_index].copy().reset_index(drop=True)


@dataclass(slots=True)
class OfflineTransitionDataset:
    root: Path
    arrays: dict[str, np.ndarray]

    @classmethod
    def load(cls, path: str | Path) -> "OfflineTransitionDataset":
        source = np.load(Path(path), allow_pickle=False)
        arrays = {key: source[key] for key in source.files}
        return cls(root=Path(path), arrays=arrays)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        np.savez_compressed(target, **self.arrays)
        return target


def _pick_first_existing(columns: list[str], frame: pd.DataFrame) -> str:
    for column in columns:
        if column in frame.columns:
            return column
    joined = ", ".join(columns)
    present = ", ".join(sorted(str(column) for column in frame.columns))
    raise ValueError(
        f"Missing required columns. Expected one of: {joined}. Present columns: {present}"
    )


def _resolve_input_paths(input_path: str | Path) -> list[Path]:
    source = Path(input_path)
    if source.is_file():
        return [source]
    if source.is_dir():
        candidates = sorted(source.glob("yellow_tripdata_*.parquet"))
        if not candidates:
            raise ValueError(
                f"No yellow_tripdata_*.parquet files were found under directory: {source}"
            )
        return candidates
    raise ValueError(f"Input path does not exist or is not supported: {source}")


def _extract_source_months(paths: list[Path]) -> list[str]:
    months: list[str] = []
    for path in paths:
        stem = path.stem
        marker = "yellow_tripdata_"
        if marker in stem:
            months.append(stem.split(marker, 1)[1])
        else:
            months.append(stem)
    return months


def _build_charge_price_proxy(time_bin_index: pd.Series) -> np.ndarray:
    fraction = time_bin_index.to_numpy(dtype=np.float32) / 144.0
    morning_peak = ((time_bin_index >= 42) & (time_bin_index <= 60)).to_numpy(dtype=np.float32)
    evening_peak = ((time_bin_index >= 96) & (time_bin_index <= 114)).to_numpy(dtype=np.float32)
    curve = 0.92 + 0.10 * np.sin(2.0 * np.pi * (fraction - 0.15))
    return (curve + 0.12 * (morning_peak + evening_peak)).astype(np.float32)


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
        "zone_mode": "grid",
        "coordinate_bounds": {
            "min_lon": float(min_lon),
            "max_lon": float(max_lon),
            "min_lat": float(min_lat),
            "max_lat": float(max_lat),
        },
    }
    return pickup_cells, dropoff_cells, metadata


def _canonicalize_orders(frame: pd.DataFrame, config: DataConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    pickup_time = _pick_first_existing(PICKUP_TIME_CANDIDATES, frame)
    dropoff_time = _pick_first_existing(DROPOFF_TIME_CANDIDATES, frame)
    fare_column = _pick_first_existing(FARE_CANDIDATES, frame)
    revenue_column = _pick_first_existing(REVENUE_CANDIDATES, frame)
    distance_column = _pick_first_existing(DISTANCE_CANDIDATES, frame)

    canonical = pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(frame[pickup_time], errors="coerce"),
            "dropoff_datetime": pd.to_datetime(frame[dropoff_time], errors="coerce"),
            "fare_amount": pd.to_numeric(frame[fare_column], errors="coerce"),
            "revenue_amount": pd.to_numeric(frame[revenue_column], errors="coerce"),
            "trip_distance_miles": pd.to_numeric(frame[distance_column], errors="coerce"),
        }
    )

    mapping_metadata: dict[str, Any]
    if all(column in frame.columns for column in LOCATION_ID_COLUMNS):
        canonical["pickup_location_id"] = pd.to_numeric(frame["PULocationID"], errors="coerce")
        canonical["dropoff_location_id"] = pd.to_numeric(frame["DOLocationID"], errors="coerce")
        canonical = canonical.dropna(
            subset=[
                "pickup_datetime",
                "dropoff_datetime",
                "fare_amount",
                "revenue_amount",
                "trip_distance_miles",
                "pickup_location_id",
                "dropoff_location_id",
            ]
        )
        canonical["pickup_location_id"] = canonical["pickup_location_id"].astype(int)
        canonical["dropoff_location_id"] = canonical["dropoff_location_id"].astype(int)
        canonical = canonical[
            canonical["pickup_location_id"].between(1, MAX_TLC_LOCATION_ID)
            & canonical["dropoff_location_id"].between(1, MAX_TLC_LOCATION_ID)
        ]
        canonical["pickup_cell"] = canonical["pickup_location_id"] - 1
        canonical["dropoff_cell"] = canonical["dropoff_location_id"] - 1
        mapping_metadata = {
            "zone_mode": "tlc_location_id",
            "location_id_to_cell": {
                str(location_id): int(location_id - 1)
                for location_id in range(1, MAX_TLC_LOCATION_ID + 1)
            },
            "cell_to_location_id": {
                str(cell): int(cell + 1) for cell in range(MAX_TLC_LOCATION_ID)
            },
        }
    elif all(column in frame.columns for column in COORDINATE_COLUMNS) and config.zone_mode == "grid":
        for column in COORDINATE_COLUMNS:
            canonical[column] = pd.to_numeric(frame[column], errors="coerce")
        canonical = canonical.dropna(
            subset=[
                "pickup_datetime",
                "dropoff_datetime",
                "fare_amount",
                "revenue_amount",
                "trip_distance_miles",
                *COORDINATE_COLUMNS,
            ]
        )
        pickup_cells, dropoff_cells, mapping_metadata = _grid_from_coordinates(canonical, config)
        canonical["pickup_location_id"] = pickup_cells + 1
        canonical["dropoff_location_id"] = dropoff_cells + 1
        canonical["pickup_cell"] = pickup_cells
        canonical["dropoff_cell"] = dropoff_cells
    else:
        present = ", ".join(sorted(str(column) for column in frame.columns))
        raise ValueError(
            "The input parquet must contain TLC location columns PULocationID/DOLocationID "
            "or coordinate columns for grid mode. Present columns: "
            f"{present}"
        )

    canonical["trip_minutes"] = (
        canonical["dropoff_datetime"] - canonical["pickup_datetime"]
    ).dt.total_seconds() / 60.0
    canonical = canonical[
        (canonical["dropoff_datetime"] > canonical["pickup_datetime"])
        & (canonical["fare_amount"] > 0)
        & (canonical["trip_distance_miles"] > 0)
        & (canonical["revenue_amount"] > 0)
    ].copy()

    canonical["trip_distance_km"] = canonical["trip_distance_miles"] * MILES_TO_KM
    canonical["service_date"] = canonical["pickup_datetime"].dt.date.astype(str)
    canonical["time_bin"] = canonical["pickup_datetime"].dt.floor(f"{config.step_minutes}min")
    canonical["time_bin_index"] = (
        canonical["pickup_datetime"].dt.hour * (60 // config.step_minutes)
        + canonical["pickup_datetime"].dt.minute // config.step_minutes
    ).astype(int)
    canonical["hour"] = canonical["pickup_datetime"].dt.hour.astype(int)

    selected_days = sorted(canonical["service_date"].unique())[: config.use_first_n_days]
    canonical = canonical[canonical["service_date"].isin(selected_days)].copy()
    start_bin = int(config.start_hour * 60 / config.step_minutes)
    end_bin = int(config.end_hour * 60 / config.step_minutes)
    canonical = canonical[
        (canonical["time_bin_index"] >= start_bin)
        & (canonical["time_bin_index"] < end_bin)
    ].copy()

    canonical["charge_price"] = _build_charge_price_proxy(canonical["time_bin_index"])
    global_trip_mean = max(float(canonical["trip_minutes"].mean()), 1e-6)
    canonical["travel_time_residual"] = (
        canonical["trip_minutes"] - global_trip_mean
    ) / global_trip_mean

    keep_columns = [
        "service_date",
        "time_bin",
        "time_bin_index",
        "pickup_datetime",
        "dropoff_datetime",
        "pickup_location_id",
        "dropoff_location_id",
        "pickup_cell",
        "dropoff_cell",
        "trip_distance_miles",
        "trip_distance_km",
        "trip_minutes",
        "fare_amount",
        "revenue_amount",
        "charge_price",
        "travel_time_residual",
    ]
    canonical = canonical[keep_columns].sort_values(
        ["service_date", "time_bin_index", "pickup_datetime", "pickup_cell", "dropoff_cell"]
    )
    return canonical.reset_index(drop=True), mapping_metadata


def _build_zone_neighbors(train_frame: pd.DataFrame, cell_count: int) -> dict[str, list[int]]:
    od_counts = (
        train_frame.groupby(["pickup_cell", "dropoff_cell"], as_index=False)
        .size()
        .sort_values(["pickup_cell", "size"], ascending=[True, False])
    )
    by_zone: dict[int, list[int]] = {zone: [] for zone in range(cell_count)}
    for record in od_counts.to_dict("records"):
        pickup_zone = int(record["pickup_cell"])
        dropoff_zone = int(record["dropoff_cell"])
        if dropoff_zone == pickup_zone:
            continue
        candidates = by_zone[pickup_zone]
        if dropoff_zone not in candidates:
            candidates.append(dropoff_zone)

    demand_rank = (
        train_frame.groupby("pickup_cell").size().sort_values(ascending=False).index.astype(int).tolist()
    )
    for zone in range(cell_count):
        candidates = by_zone[zone]
        for fallback_zone in demand_rank:
            if fallback_zone != zone and fallback_zone not in candidates:
                candidates.append(int(fallback_zone))
            if len(candidates) >= ACTIONS_PER_ZONE:
                break
        if not candidates:
            candidates.append(zone)
        while len(candidates) < ACTIONS_PER_ZONE:
            candidates.append(candidates[-1])
        by_zone[zone] = candidates[:ACTIONS_PER_ZONE]
    return {str(zone): neighbors for zone, neighbors in by_zone.items()}


def _build_metadata(
    processed: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    config: DataConfig,
    mapping_metadata: dict[str, Any],
    charge_station_count: int,
    source_files: list[Path],
) -> dict[str, Any]:
    train_frame = split_frames["train"]
    cell_count = int(config.cell_count if config.zone_mode == "grid" else MAX_TLC_LOCATION_ID)
    demand_totals = (
        train_frame.groupby("pickup_cell").size().reindex(range(cell_count), fill_value=0)
    )
    demand_priors = (demand_totals / max(float(demand_totals.sum()), 1.0)).round(8)
    charge_stations = (
        demand_totals.sort_values(ascending=False)
        .head(min(charge_station_count, cell_count))
        .index.astype(int)
        .tolist()
    )
    zone_neighbors = _build_zone_neighbors(train_frame, cell_count)
    split_dates = {
        split: sorted(frame["service_date"].unique().tolist())
        for split, frame in split_frames.items()
    }
    od_summary = (
        train_frame.groupby(["pickup_cell", "dropoff_cell"], as_index=False)
        .agg(
            order_count=("pickup_cell", "size"),
            mean_trip_minutes=("trip_minutes", "mean"),
            mean_trip_distance_km=("trip_distance_km", "mean"),
            mean_revenue_amount=("revenue_amount", "mean"),
        )
        .sort_values("order_count", ascending=False)
        .head(4096)
        .to_dict("records")
    )
    demand_summary = (
        train_frame.groupby(["pickup_cell", "time_bin_index"], as_index=False)
        .size()
        .rename(columns={"size": "order_count"})
        .to_dict("records")
    )
    charge_price_summary = {
        "mean": float(train_frame["charge_price"].mean()),
        "std": float(train_frame["charge_price"].std(ddof=0) or 0.0),
        "min": float(train_frame["charge_price"].min()),
        "max": float(train_frame["charge_price"].max()),
    }
    trip_duration_summary = {
        "mean": float(processed["trip_minutes"].mean()),
        "std": float(processed["trip_minutes"].std(ddof=0) or 0.0),
        "p50": float(processed["trip_minutes"].quantile(0.5)),
        "p90": float(processed["trip_minutes"].quantile(0.9)),
    }
    trip_distance_summary = {
        "mean_km": float(processed["trip_distance_km"].mean()),
        "std_km": float(processed["trip_distance_km"].std(ddof=0) or 0.0),
        "p50_km": float(processed["trip_distance_km"].quantile(0.5)),
        "p90_km": float(processed["trip_distance_km"].quantile(0.9)),
    }
    fare_summary = {
        "mean_fare_amount": float(processed["fare_amount"].mean()),
        "mean_revenue_amount": float(processed["revenue_amount"].mean()),
        "std_revenue_amount": float(processed["revenue_amount"].std(ddof=0) or 0.0),
    }
    mapping_metadata["zone_neighbors"] = zone_neighbors
    source_months = _extract_source_months(source_files)
    available_days = sorted(processed["service_date"].unique().tolist())
    split_day_counts = {split: len(days) for split, days in split_dates.items()}
    split_date_ranges = {
        split: {
            "start": days[0] if days else None,
            "end": days[-1] if days else None,
        }
        for split, days in split_dates.items()
    }
    return {
        "data": {
            "zone_mode": mapping_metadata.get("zone_mode", config.zone_mode),
            "grid_rows": config.grid_rows,
            "grid_cols": config.grid_cols,
            "cell_count": cell_count,
            "step_minutes": config.step_minutes,
            "start_hour": config.start_hour,
            "end_hour": config.end_hour,
            "episode_steps": config.episode_steps,
            "time_bins_per_day": config.time_bins_per_day,
        },
        "source_files": [str(path) for path in source_files],
        "source_months": source_months,
        "available_days": available_days,
        "split_day_counts": split_day_counts,
        "split_date_ranges": split_date_ranges,
        "splits": split_dates,
        "charge_stations": charge_stations,
        "zone_demand_priors": demand_priors.tolist(),
        "global_defaults": {
            "mean_trip_minutes": float(processed["trip_minutes"].mean()),
            "mean_fare": float(processed["revenue_amount"].mean()),
            "mean_trip_distance_km": float(processed["trip_distance_km"].mean()),
            "mean_charge_price": float(processed["charge_price"].mean()),
        },
        "calibration": {
            "od_travel_time_summary": od_summary,
            "demand_per_zone_time_summary": demand_summary,
            "trip_distance_summary": trip_distance_summary,
            "trip_duration_summary": trip_duration_summary,
            "fare_summary": fare_summary,
            "charge_price_summary": charge_price_summary,
            "charger_metadata_placeholder": [
                {
                    "station_zone": int(zone),
                    "capacity": 4,
                    "queue_slots": 8,
                }
                for zone in charge_stations
            ],
            "notes": {
                "charge_price": "NYC TLC does not include electricity prices. The project uses a time-of-day proxy for charging cost.",
                "reposition_graph": "For TLC zone mode, move actions are mapped to four data-driven neighboring zones derived from historical OD transitions.",
            },
        },
        "mapping": mapping_metadata,
    }


def prepare_nyc_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    config: DataConfig,
    charge_station_count: int = 5,
) -> PreparedDataset:
    source_files = _resolve_input_paths(input_path)
    source = pd.concat(
        [pd.read_parquet(path) for path in source_files],
        ignore_index=True,
    )
    processed, mapping_metadata = _canonicalize_orders(source, config)

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
    split_frames = {
        split: processed[processed["service_date"].isin(days)].copy().reset_index(drop=True)
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
        source_files=source_files,
    )
    dump_json(metadata, dataset_root / "metadata.json")
    return PreparedDataset(root=dataset_root, metadata=metadata, splits=split_frames)


def fit_normalization_statistics(dataset: PreparedDataset) -> dict[str, list[float] | float]:
    train_frame = dataset.splits["train"].copy()
    demand_series = (
        train_frame.groupby("pickup_cell").size().reindex(
            range(dataset.metadata["data"]["cell_count"]),
            fill_value=0.0,
        )
    )
    temporal_scalars = train_frame[["charge_price", "travel_time_residual"]].to_numpy(dtype=np.float32)
    return {
        "demand_mean": demand_series.astype(float).tolist(),
        "demand_std": np.sqrt(np.maximum(demand_series.to_numpy(dtype=np.float32), 1e-6)).tolist(),
        "charge_price_mean": float(temporal_scalars[:, 0].mean()),
        "charge_price_std": float(temporal_scalars[:, 0].std() or 1.0),
        "travel_time_residual_mean": float(temporal_scalars[:, 1].mean()),
        "travel_time_residual_std": float(temporal_scalars[:, 1].std() or 1.0),
        "trip_distance_km_mean": float(train_frame["trip_distance_km"].mean()),
        "trip_distance_km_std": float(train_frame["trip_distance_km"].std(ddof=0) or 1.0),
        "trip_minutes_mean": float(train_frame["trip_minutes"].mean()),
        "trip_minutes_std": float(train_frame["trip_minutes"].std(ddof=0) or 1.0),
    }


def export_offline_transition_dataset(
    dataset: PreparedDataset,
    output_path: str | Path,
    config: Any,
    episodes: int | None = None,
    seed: int | None = None,
) -> OfflineTransitionDataset:
    from .baselines import GreedyDispatchPolicy
    from .env import CometTaxiEnv

    env = CometTaxiEnv(dataset, config, seed=seed if seed is not None else config.train.seed)
    policy = GreedyDispatchPolicy(config)
    episode_count = int(episodes if episodes is not None else config.offline_rl.dataset_episodes)

    local_obs: list[np.ndarray] = []
    fleet_signature: list[np.ndarray] = []
    temporal_history: list[np.ndarray] = []
    agent_mask: list[np.ndarray] = []
    action_mask: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[float] = []
    cost_vectors: list[np.ndarray] = []
    next_local_obs: list[np.ndarray] = []
    next_fleet_signature: list[np.ndarray] = []
    next_temporal_history: list[np.ndarray] = []
    next_agent_mask: list[np.ndarray] = []
    next_action_mask: list[np.ndarray] = []
    next_rewards_to_go: list[float] = []
    aux_next_demand: list[np.ndarray] = []
    aux_charger_occupancy: list[np.ndarray] = []
    aux_travel_residual: list[np.ndarray] = []

    for _ in range(episode_count):
        observation = env.reset("train")
        episode_rewards: list[float] = []
        done = False
        while not done:
            action = policy.act(observation)
            next_observation, _, team_reward, done, info = env.step(action)
            episode_rewards.append(team_reward)
            local_obs.append(observation["local_obs"])
            fleet_signature.append(observation["fleet_signature"])
            temporal_history.append(observation["temporal_history"])
            agent_mask.append(observation["agent_mask"])
            action_mask.append(observation["action_mask"])
            actions.append(action)
            rewards.append(team_reward)
            dones.append(float(done))
            cost_vectors.append(
                np.asarray(
                    [
                        info["costs"]["battery_violation_cost"],
                        info["costs"]["charger_overflow_cost"],
                        info["costs"]["service_violation_cost"],
                    ],
                    dtype=np.float32,
                )
            )
            next_local_obs.append(next_observation["local_obs"])
            next_fleet_signature.append(next_observation["fleet_signature"])
            next_temporal_history.append(next_observation["temporal_history"])
            next_agent_mask.append(next_observation["agent_mask"])
            next_action_mask.append(next_observation["action_mask"])
            aux_next_demand.append(info["aux_targets"]["next_demand"])
            aux_charger_occupancy.append(info["aux_targets"]["charger_occupancy"])
            aux_travel_residual.append(
                np.asarray([info["aux_targets"]["travel_time_residual"]], dtype=np.float32)
            )
            observation = next_observation

        running_return = 0.0
        episode_returns: list[float] = []
        for reward in reversed(episode_rewards):
            running_return = float(reward) + config.train.gamma * running_return
            episode_returns.append(running_return)
        next_rewards_to_go.extend(reversed(episode_returns))

    arrays = {
        "local_obs": np.stack(local_obs).astype(np.float32),
        "fleet_signature": np.stack(fleet_signature).astype(np.float32),
        "temporal_history": np.stack(temporal_history).astype(np.float32),
        "agent_mask": np.stack(agent_mask).astype(np.float32),
        "action_mask": np.stack(action_mask).astype(np.float32),
        "actions": np.stack(actions).astype(np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "returns": np.asarray(next_rewards_to_go, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "costs": np.stack(cost_vectors).astype(np.float32),
        "next_local_obs": np.stack(next_local_obs).astype(np.float32),
        "next_fleet_signature": np.stack(next_fleet_signature).astype(np.float32),
        "next_temporal_history": np.stack(next_temporal_history).astype(np.float32),
        "next_agent_mask": np.stack(next_agent_mask).astype(np.float32),
        "next_action_mask": np.stack(next_action_mask).astype(np.float32),
        "aux_next_demand": np.stack(aux_next_demand).astype(np.float32),
        "aux_charger_occupancy": np.stack(aux_charger_occupancy).astype(np.float32),
        "aux_travel_time_residual": np.stack(aux_travel_residual).astype(np.float32),
    }
    offline = OfflineTransitionDataset(root=Path(output_path), arrays=arrays)
    offline.save(output_path)
    return offline
