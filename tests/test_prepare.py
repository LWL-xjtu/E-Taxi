from __future__ import annotations

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
        "pickup_cell",
        "dropoff_cell",
        "demand_count",
        "mean_trip_minutes",
        "mean_fare",
        "charge_price",
        "travel_time_residual",
    }
    assert expected_columns.issubset(dataset.splits["train"].columns)
    assert len(dataset.metadata["charge_stations"]) == config.env.charge_station_count
    assert "calibration" in dataset.metadata
    stats = fit_normalization_statistics(dataset)
    assert "charge_price_mean" in stats
