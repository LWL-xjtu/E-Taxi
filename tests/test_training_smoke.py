from __future__ import annotations

from pathlib import Path

from comet_taxi.config import load_experiment_config
from comet_taxi.data import prepare_nyc_dataset
from comet_taxi.synthetic import write_synthetic_parquet
from comet_taxi.trainer import CometTrainer


def test_smoke_training_produces_outputs(tmp_path: Path) -> None:
    config = load_experiment_config("configs/smoke.toml")
    parquet_path = tmp_path / "synthetic.parquet"
    data_dir = tmp_path / "prepared"
    output_dir = tmp_path / "outputs"

    write_synthetic_parquet(parquet_path, rows_per_step=4)
    prepare_nyc_dataset(
        parquet_path,
        data_dir,
        config.data,
        charge_station_count=config.env.charge_station_count,
    )

    trainer = CometTrainer(config, data_dir, output_dir)
    history = trainer.train()

    assert not history.empty
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "reward_curve.csv").exists()
    checkpoints = list((output_dir / "checkpoints").glob("*.pt"))
    assert checkpoints
