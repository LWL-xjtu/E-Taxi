from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from comet_taxi.config import load_experiment_config
from comet_taxi.data import prepare_nyc_dataset
from comet_taxi.synthetic import write_synthetic_parquet
from comet_taxi.trainer import CometTrainer


def _build_small_formal_config():
    config = deepcopy(load_experiment_config("configs/formal_generalist.toml"))
    config.data.use_first_n_days = 7
    config.data.train_days = 5
    config.data.val_days = 1
    config.data.test_days = 1
    config.train.total_episodes = 1
    config.train.eval_interval = 1
    config.train.save_interval = 1
    config.train.ppo_epochs = 1
    config.train.minibatch_size = 8
    config.offline_rl.dataset_episodes = 1
    config.offline_rl.pretrain_epochs = 1
    config.offline_rl.batch_size = 8
    config.online_finetune.online_replay_capacity = 256
    return config


def test_smoke_training_produces_outputs_and_resume_checkpoint(tmp_path: Path) -> None:
    config = _build_small_formal_config()
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
    assert (output_dir / "checkpoints" / "latest.pt").exists()
    assert (output_dir / "checkpoints" / "best_val.pt").exists()
    metrics = pd.read_csv(output_dir / "metrics.csv")
    for column in (
        "train_battery_violation_rate",
        "train_charger_overflow_rate",
        "train_service_violation_rate",
        "lambda_battery",
        "lambda_charger",
        "lambda_service",
        "train_fallback_rate",
        "train_uncertainty_trigger_rate",
        "offline_ratio",
        "online_ratio",
        "wall_clock_seconds",
        "episodes_per_hour",
        "steps_per_second",
        "best_val_mean_team_reward",
        "checkpoint_tag",
    ):
        assert column in metrics.columns

    resumed_config = _build_small_formal_config()
    resumed_config.train.total_episodes = 2
    resumed_trainer = CometTrainer(
        resumed_config,
        data_dir,
        output_dir,
        resume_checkpoint=output_dir / "checkpoints" / "latest.pt",
    )
    resumed_history = resumed_trainer.train()
    assert resumed_trainer.start_episode == 2
    assert int(resumed_history["episode"].max()) == 2
