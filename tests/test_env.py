from __future__ import annotations

from pathlib import Path

import numpy as np

from comet_taxi.config import load_experiment_config
from comet_taxi.constants import ACTION_GO_CHARGE
from comet_taxi.data import PreparedDataset, prepare_nyc_dataset
from comet_taxi.env import CometTaxiEnv
from comet_taxi.synthetic import write_synthetic_parquet


def _build_env(tmp_path: Path) -> CometTaxiEnv:
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def test_env_reset_and_step_return_v2_observation(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    observation = env.reset("train")
    assert observation["local_obs"].shape == (env.config.env.nmax, 10)
    assert observation["temporal_history"].shape[0] == env.config.temporal.history_len
    assert observation["agent_mask"].shape == (env.config.env.nmax,)
    assert observation["action_mask"].shape == (env.config.env.nmax, 7)

    actions = np.zeros(env.config.env.nmax, dtype=np.int64)
    _, per_agent_reward, team_reward, _, info = env.step(actions)
    assert per_agent_reward.shape == (env.config.env.nmax,)
    assert isinstance(info["costs"], dict)
    assert set(info["costs"].keys()) == {
        "battery_violation_cost",
        "charger_overflow_cost",
        "service_violation_cost",
    }
    active_rewards = per_agent_reward[observation["agent_mask"] > 0]
    assert np.isclose(team_reward, float(active_rewards.mean()))


def test_charger_overflow_logic_is_exposed(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    observation = env.reset("train")
    charge_zone = env.charge_stations[0]
    for vehicle in env.vehicles:
        vehicle.zone = charge_zone
        vehicle.mode = 0
        vehicle.soc = 0.1
    observation = env._build_observation()
    actions = np.full(env.config.env.nmax, ACTION_GO_CHARGE, dtype=np.int64)
    _, _, _, _, info = env.step(actions)
    assert info["costs"]["charger_overflow_cost"] >= 0.0
    assert info["charger_overflow_rate"] >= 0.0


def test_domain_randomization_is_reproducible(tmp_path: Path) -> None:
    env_a = _build_env(tmp_path / "a")
    env_b = _build_env(tmp_path / "b")
    env_a.reset("train")
    env_b.reset("train")
    assert np.isclose(env_a.current_demand_scale, env_b.current_demand_scale)
    assert np.isclose(env_a.current_travel_noise, env_b.current_travel_noise)


def test_history_buffer_updates(tmp_path: Path) -> None:
    env = _build_env(tmp_path)
    observation = env.reset("train")
    before = observation["temporal_history"].copy()
    env.step(np.zeros(env.config.env.nmax, dtype=np.int64))
    after = env._build_observation()["temporal_history"]
    assert not np.array_equal(before, after)
