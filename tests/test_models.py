from __future__ import annotations

from pathlib import Path

import torch

from comet_taxi.config import load_experiment_config
from comet_taxi.data import PreparedDataset, prepare_nyc_dataset
from comet_taxi.models import (
    COMETActorV2,
    CostCritic,
    EnsembleCritic,
    FleetSetEncoder,
    TemporalEncoder,
    VehicleTokenEncoder,
    infer_model_dimensions,
)
from comet_taxi.synthetic import write_synthetic_parquet


def _build_dims(tmp_path: Path):
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
    dims = infer_model_dimensions(
        dataset.metadata["data"]["cell_count"],
        charger_count=config.env.charge_station_count,
        history_len=config.temporal.history_len,
    )
    return config, dims


def test_actor_v2_and_critic_shapes(tmp_path: Path) -> None:
    config, dims = _build_dims(tmp_path)
    actor = COMETActorV2(config.model, config.set_encoder, config.temporal, dims)
    critic = EnsembleCritic(config.model, config.set_encoder, config.temporal, dims)
    cost_critic = CostCritic(config.model, config.set_encoder, config.temporal, dims)

    batch_size = 2
    local_obs = torch.zeros(batch_size, config.env.nmax, dims.local_dim)
    fleet_signature = torch.zeros(batch_size, dims.fleet_signature_dim)
    temporal_history = torch.zeros(batch_size, config.temporal.history_len, dims.temporal_feature_dim)
    agent_mask = torch.zeros(batch_size, config.env.nmax)
    agent_mask[:, :8] = 1.0
    action_mask = torch.ones(batch_size, config.env.nmax, dims.action_dim)

    observation = {
        "local_obs": local_obs,
        "fleet_signature": fleet_signature,
        "temporal_history": temporal_history,
        "agent_mask": agent_mask,
        "action_mask": action_mask,
    }
    actor_output = actor(observation)
    critic_output = critic(observation)
    cost_output = cost_critic(observation)

    assert actor_output.logits.shape == (batch_size, config.env.nmax, dims.action_dim)
    assert actor_output.aux_predictions["next_demand"].shape == (batch_size, dims.cell_count)
    assert actor_output.aux_predictions["charger_occupancy"].shape == (
        batch_size,
        config.env.charge_station_count,
    )
    assert critic_output.mean.shape == (batch_size,)
    assert critic_output.ensemble_values.shape[-1] == config.model.critic_ensemble_size
    assert set(cost_output.keys()) == {
        "battery_violation_cost",
        "charger_overflow_cost",
        "service_violation_cost",
    }


def test_set_encoder_is_permutation_invariant(tmp_path: Path) -> None:
    config, dims = _build_dims(tmp_path)
    token_encoder = VehicleTokenEncoder(config.model, dims)
    set_encoder = FleetSetEncoder(config.set_encoder, config.model, dims)

    local_obs = torch.zeros(1, config.env.nmax, dims.local_dim)
    local_obs[0, :4, 0] = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    local_obs[0, :4, 4] = torch.tensor([0.2, 0.4, 0.7, 0.9], dtype=torch.float32)
    agent_mask = torch.zeros(1, config.env.nmax)
    agent_mask[0, :4] = 1.0
    fleet_signature = torch.zeros(1, dims.fleet_signature_dim)

    tokens = token_encoder(local_obs)
    permuted_indices = torch.tensor([2, 0, 3, 1] + list(range(4, config.env.nmax)))
    permuted_tokens = tokens[:, permuted_indices]
    permuted_mask = agent_mask[:, permuted_indices]

    original = set_encoder(tokens, agent_mask, fleet_signature)
    permuted = set_encoder(permuted_tokens, permuted_mask, fleet_signature)

    assert torch.allclose(original, permuted, atol=1e-5)


def test_temporal_encoder_supports_history_shapes(tmp_path: Path) -> None:
    config, dims = _build_dims(tmp_path)
    temporal_encoder = TemporalEncoder(config.temporal, config.model, dims)
    history = torch.randn(3, config.temporal.history_len, dims.temporal_feature_dim)
    encoded = temporal_encoder(history)
    assert encoded.shape == (3, config.temporal.hidden_dim)
