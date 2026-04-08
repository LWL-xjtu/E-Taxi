from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from comet_taxi.config import load_experiment_config
from comet_taxi.data import fit_normalization_statistics, prepare_nyc_dataset
from comet_taxi.synthetic import write_synthetic_parquet


def test_prepare_nyc_dataset_outputs_expected_files(tmp_path: Path) -> None:
    config = load_experiment_config("configs/smoke.toml")
    parquet_path = tmp_path / "synthetic.parquet"
    output_dir = tmp_path / "prepared"

    write_synthetic_parquet(parquet_path, rows_per_step=6)
    dataset = prepare_nyc_dataset(
        parquet_path,
        output_dir,
        config.data,
        charge_station_count=config.env.charge_station_count,
    )

    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "val.parquet").exists()
    assert (output_dir / "test.parquet").exists()
    assert (output_dir / "metadata.json").exists()
    assert set(dataset.splits.keys()) == {"train", "val", "test"}

    expected_columns = {
        "service_date",
        "time_bin",
        "time_bin_index",
        "pickup_location_id",
        "dropoff_location_id",
        "pickup_cell",
        "dropoff_cell",
        "trip_distance_km",
        "trip_minutes",
        "fare_amount",
        "revenue_amount",
        "charge_price",
        "travel_time_residual",
    }
    assert expected_columns.issubset(dataset.splits["train"].columns)
    assert len(dataset.metadata["charge_stations"]) == config.env.charge_station_count
    assert "calibration" in dataset.metadata
    assert dataset.metadata["data"]["zone_mode"] == "tlc_location_id"
    first_day = dataset.metadata["splits"]["train"][0]
    day_orders = dataset.get_day_orders(first_day, split="train")
    assert not day_orders.empty
    bin_orders = dataset.get_orders_for_day_and_bin(first_day, 0, split="train")
    assert isinstance(bin_orders, type(day_orders))
    stats = fit_normalization_statistics(dataset)
    assert "charge_price_mean" in stats
    assert "trip_distance_km_mean" in stats


def test_prepare_nyc_dataset_accepts_directory_and_writes_formal_metadata(tmp_path: Path) -> None:
    config = deepcopy(load_experiment_config("configs/formal_generalist.toml"))
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "prepared"
    raw_dir.mkdir(parents=True, exist_ok=True)

    write_synthetic_parquet(
        raw_dir / "yellow_tripdata_2025-12.parquet",
        start="2025-12-01 00:00:00",
        days=31,
        rows_per_step=1,
    )
    write_synthetic_parquet(
        raw_dir / "yellow_tripdata_2026-01.parquet",
        start="2026-01-01 00:00:00",
        days=31,
        rows_per_step=1,
    )
    write_synthetic_parquet(
        raw_dir / "yellow_tripdata_2026-02.parquet",
        start="2026-02-01 00:00:00",
        days=28,
        rows_per_step=1,
    )

    dataset = prepare_nyc_dataset(
        raw_dir,
        output_dir,
        config.data,
        charge_station_count=config.env.charge_station_count,
    )

    metadata = dataset.metadata
    assert metadata["source_months"] == ["2025-12", "2026-01", "2026-02"]
    assert metadata["split_day_counts"] == {"train": 70, "val": 10, "test": 10}
    assert len(metadata["available_days"]) == 90
    assert metadata["split_date_ranges"]["train"]["start"] == "2025-12-01"
    assert metadata["split_date_ranges"]["train"]["end"] == "2026-02-08"
    assert metadata["split_date_ranges"]["val"]["start"] == "2026-02-09"
    assert metadata["split_date_ranges"]["val"]["end"] == "2026-02-18"
    assert metadata["split_date_ranges"]["test"]["start"] == "2026-02-19"
    assert metadata["split_date_ranges"]["test"]["end"] == "2026-02-28"
    assert len(metadata["source_files"]) == 3
