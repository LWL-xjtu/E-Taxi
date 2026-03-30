from __future__ import annotations

from pathlib import Path

import numpy as np

from comet_taxi.config import load_experiment_config
from comet_taxi.data import PreparedDataset, prepare_nyc_dataset
from comet_taxi.env import CometTaxiEnv
from comet_taxi.synthetic import write_synthetic_parquet


def _build_env(tmp_path: Path) -> CometTaxiEnv:
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
    return CometTaxiEnv(dataset, config, seed=3)


def test_env_returns_fixed_shapes(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    observation = env.reset("train")
    assert observation["local_obs"].shape == (env.config.env.nmax, 10)
    assert observation["agent_mask"].shape == (env.config.env.nmax,)
    assert observation["action_mask"].shape == (env.config.env.nmax, 7)
    assert observation["fleet_signature"].ndim == 1


def test_team_reward_uses_real_agents_only(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    observation = env.reset("train")
    actions = np.zeros(env.config.env.nmax, dtype=np.int64)
    _, per_agent_reward, team_reward, _, _ = env.step(actions)
    active_rewards = per_agent_reward[observation["agent_mask"] > 0]
    assert np.isclose(team_reward, float(active_rewards.mean()))


def test_ghost_slots_have_no_valid_actions(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    observation = env.reset("train")
    ghost_slots = np.where(observation["agent_mask"] == 0)[0]
    if len(ghost_slots) > 0:
        assert np.all(observation["action_mask"][ghost_slots] == 0)
