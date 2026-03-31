from __future__ import annotations

from pathlib import Path

from comet_taxi.config import load_experiment_config
from comet_taxi.data import PreparedDataset, prepare_nyc_dataset
from comet_taxi.env import CometTaxiEnv
from comet_taxi.evaluation import evaluate_greedy
from comet_taxi.synthetic import write_synthetic_parquet


def test_stress_eval_pipeline_writes_expected_columns(tmp_path: Path) -> None:
    config = load_experiment_config("configs/smoke.toml")
    parquet_path = tmp_path / "synthetic.parquet"
    data_dir = tmp_path / "prepared"
    output_dir = tmp_path / "eval"
    write_synthetic_parquet(parquet_path, rows_per_step=4)
    prepare_nyc_dataset(
        parquet_path,
        data_dir,
        config.data,
        charge_station_count=config.env.charge_station_count,
    )
    dataset = PreparedDataset.load(data_dir)
    env = CometTaxiEnv(dataset, config, seed=config.train.seed)
    metrics = evaluate_greedy(config, env, output_dir, split="test", episodes=1, stress=True)
    assert not metrics.empty
    for column in (
        "scenario",
        "mean_team_reward",
        "battery_violation_rate",
        "charger_overflow_rate",
        "service_violation_rate",
        "fallback_rate",
        "uncertainty_trigger_rate",
    ):
        assert column in metrics.columns


def test_ten_to_fifteen_config_exposes_unseen_fleet_15(tmp_path: Path) -> None:
    config = load_experiment_config("configs/ten_to_fifteen_real.toml")
    parquet_path = tmp_path / "synthetic.parquet"
    data_dir = tmp_path / "prepared"
    output_dir = tmp_path / "eval"
    write_synthetic_parquet(parquet_path, rows_per_step=4)
    prepare_nyc_dataset(
        parquet_path,
        data_dir,
        config.data,
        charge_station_count=config.env.charge_station_count,
    )
    dataset = PreparedDataset.load(data_dir)
    env = CometTaxiEnv(dataset, config, seed=config.train.seed)
    metrics = evaluate_greedy(config, env, output_dir, split="test", episodes=1, stress=True)
    assert "unseen_fleet_15" in set(metrics["scenario"].tolist())
