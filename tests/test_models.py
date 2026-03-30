from __future__ import annotations

from pathlib import Path

import torch

from comet_taxi.config import load_experiment_config
from comet_taxi.data import PreparedDataset, prepare_nyc_dataset
from comet_taxi.models import COMETActor, COMETCritic, infer_model_dimensions
from comet_taxi.synthetic import write_synthetic_parquet


def test_actor_and_critic_forward_shapes(tmp_path: Path) -> None:
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
    dims = infer_model_dimensions(dataset.metadata["data"]["cell_count"])

    actor = COMETActor(config.model, dims)
    critic = COMETCritic(config.model, dims)

    batch_size = 2
    local_obs = torch.zeros(batch_size, config.env.nmax, 10)
    fleet_signature = torch.zeros(batch_size, dims.fleet_signature_dim)
    agent_mask = torch.ones(batch_size, config.env.nmax)

    logits, aux_prediction = actor(local_obs, fleet_signature)
    values = critic(local_obs, fleet_signature, agent_mask)

    assert logits.shape == (batch_size, config.env.nmax, 7)
    assert aux_prediction.shape == (batch_size, dims.cell_count)
    assert values.shape == (batch_size,)
