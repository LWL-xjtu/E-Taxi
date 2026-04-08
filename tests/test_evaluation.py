from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from comet_taxi.config import load_experiment_config
from comet_taxi.data import PreparedDataset, prepare_nyc_dataset
from comet_taxi.env import CometTaxiEnv
from comet_taxi.evaluation import evaluate_checkpoint, evaluate_greedy
from comet_taxi.models import COMETActorV2, CostCritic, EnsembleCritic, ObservationNormalizer, infer_model_dimensions
from comet_taxi.synthetic import write_synthetic_parquet
from comet_taxi.trainer import CometTrainer


def test_stress_eval_pipeline_writes_expected_columns(tmp_path: Path) -> None:
    config = deepcopy(load_experiment_config("configs/generalization_eval.toml"))
    config.data.use_first_n_days = 7
    config.data.train_days = 5
    config.data.val_days = 1
    config.data.test_days = 1
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
        "execution_mode",
        "mean_team_reward",
        "battery_violation_rate",
        "charger_overflow_rate",
        "service_violation_rate",
        "service_utilization_rate",
        "fallback_rate",
        "uncertainty_trigger_rate",
        "data_window",
        "model_variant",
    ):
        assert column in metrics.columns
    assert set(metrics["execution_mode"]) == {"greedy"}


def test_generalization_eval_config_exposes_full_scenario_matrix(tmp_path: Path) -> None:
    config = deepcopy(load_experiment_config("configs/generalization_eval.toml"))
    config.data.use_first_n_days = 7
    config.data.train_days = 5
    config.data.val_days = 1
    config.data.test_days = 1
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
    scenarios = set(metrics["scenario"].tolist())
    expected = {
        "standard_test",
        "unseen_fleet_8",
        "unseen_fleet_16",
        "unseen_fleet_24",
        "unseen_fleet_32",
        "unseen_fleet_48",
        "unseen_fleet_64",
        "charger_outage_0.25",
        "charger_outage_0.50",
        "demand_shock_1.25",
        "demand_shock_1.50",
        "travel_time_1.15",
        "travel_time_1.30",
        "mixed_ood_stress",
    }
    assert expected.issubset(scenarios)


def test_checkpoint_evaluation_outputs_policy_and_runtime_modes(tmp_path: Path) -> None:
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
    config.online_finetune.online_replay_capacity = 128

    parquet_path = tmp_path / "synthetic.parquet"
    data_dir = tmp_path / "prepared"
    train_output_dir = tmp_path / "train"
    eval_output_dir = tmp_path / "eval"
    write_synthetic_parquet(parquet_path, rows_per_step=4)
    prepare_nyc_dataset(
        parquet_path,
        data_dir,
        config.data,
        charge_station_count=config.env.charge_station_count,
    )
    trainer = CometTrainer(config, data_dir, train_output_dir)
    trainer.train()

    dataset = PreparedDataset.load(data_dir)
    env = CometTaxiEnv(dataset, config, seed=config.train.seed)
    dims = infer_model_dimensions(
        dataset.metadata["data"]["cell_count"],
        charger_count=len(dataset.metadata["charge_stations"]),
        history_len=config.temporal.history_len,
    )
    actor = COMETActorV2(config.model, config.set_encoder, config.temporal, dims)
    critic = EnsembleCritic(config.model, config.set_encoder, config.temporal, dims)
    cost_critic = CostCritic(config.model, config.set_encoder, config.temporal, dims)
    normalizer = ObservationNormalizer(dims)
    metrics = evaluate_checkpoint(
        config=config,
        env=env,
        actor=actor,
        checkpoint_path=train_output_dir / "checkpoints" / "best_policy_val.pt",
        output_dir=eval_output_dir,
        split="test",
        episodes=1,
        critic=critic,
        cost_critic=cost_critic,
        normalizer=normalizer,
        stress=False,
    )
    assert {"policy_only", "planner_runtime"}.issubset(set(metrics["execution_mode"]))
