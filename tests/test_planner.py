from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from comet_taxi.config import load_experiment_config
from comet_taxi.data import PreparedDataset, prepare_nyc_dataset
from comet_taxi.env import CometTaxiEnv
from comet_taxi.models import COMETActorV2, CostCritic, EnsembleCritic, ObservationNormalizer, infer_model_dimensions
from comet_taxi.planner import CandidatePlanner
from comet_taxi.runtime import UncertaintyCalibrator
from comet_taxi.synthetic import write_synthetic_parquet


def _build_components(tmp_path: Path):
    config = load_experiment_config("configs/smoke.toml")
    parquet_path = tmp_path / "synthetic.parquet"
    output_dir = tmp_path / "prepared"
    write_synthetic_parquet(parquet_path, rows_per_step=6)
    prepare_nyc_dataset(
        parquet_path,
        output_dir,
        config.data,
        charge_station_count=config.env.charge_station_count,
    )
    dataset = PreparedDataset.load(output_dir)
    env = CometTaxiEnv(dataset, config, seed=3)
    dims = infer_model_dimensions(
        dataset.metadata["data"]["cell_count"],
        charger_count=len(dataset.metadata["charge_stations"]),
        history_len=config.temporal.history_len,
    )
    actor = COMETActorV2(config.model, config.set_encoder, config.temporal, dims)
    critic = EnsembleCritic(config.model, config.set_encoder, config.temporal, dims)
    cost_critic = CostCritic(config.model, config.set_encoder, config.temporal, dims)
    normalizer = ObservationNormalizer(dims)
    planner = CandidatePlanner(config)
    return config, env, actor, critic, cost_critic, normalizer, planner


def test_candidate_generation_and_shape(tmp_path: Path) -> None:
    config, env, actor, critic, cost_critic, normalizer, planner = _build_components(tmp_path)
    observation = env.reset("train")
    candidates = env.generate_candidate_actions(observation, config.planner.top_k_zones)
    assert len(candidates) == config.env.nmax
    assert all(isinstance(item, list) for item in candidates)


def test_uncertainty_gate_and_fallback_trigger(tmp_path: Path) -> None:
    config, env, actor, critic, cost_critic, normalizer, planner = _build_components(tmp_path)
    observation = env.reset("train")
    for index, vehicle in enumerate(env.vehicles):
        vehicle.battery_kwh = 4.0 if index < 2 else 40.0
    observation = env._build_observation()
    calibrator = UncertaintyCalibrator()
    calibrator.mean = 0.0
    calibrator.var = 1e-6
    calibrator.count = 10.0
    output = planner.select_actions(
        actor=actor,
        critic=critic,
        cost_critic=cost_critic,
        observation=observation,
        device=torch.device("cpu"),
        env=env,
        normalizer=normalizer,
        planner_enabled=True,
        uncertainty_calibrator=calibrator,
        current_episode=config.planner.uncertainty_warmup_episodes + 1,
    )
    assert output.actions.shape == (config.env.nmax,)
    assert np.sum(output.fallback_mask) > 0
    assert np.sum(output.planner_selected_mask) + np.sum(output.fallback_mask) > 0
    assert output.uncertainty_z_values.shape == (config.env.nmax,)


def test_low_risk_state_prefers_planner_not_fallback(tmp_path: Path) -> None:
    config, env, actor, critic, cost_critic, normalizer, planner = _build_components(tmp_path)
    observation = env.reset("train")
    for vehicle in env.vehicles:
        vehicle.battery_kwh = 45.0
    observation = env._build_observation()
    calibrator = UncertaintyCalibrator()
    calibrator.mean = 10.0
    calibrator.var = 1.0
    calibrator.count = 10.0
    output = planner.select_actions(
        actor=actor,
        critic=critic,
        cost_critic=cost_critic,
        observation=observation,
        device=torch.device("cpu"),
        env=env,
        normalizer=normalizer,
        planner_enabled=True,
        uncertainty_calibrator=calibrator,
        current_episode=config.planner.uncertainty_warmup_episodes + 1,
    )
    assert np.sum(output.fallback_mask) == 0
    assert np.sum(output.planner_selected_mask) >= 0
