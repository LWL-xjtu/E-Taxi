from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig
from .constants import ACTION_DIM, LOCAL_OBS_DIM


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )


@dataclass(slots=True)
class ModelDimensions:
    local_dim: int = LOCAL_OBS_DIM
    action_dim: int = ACTION_DIM
    cell_count: int = 64
    fleet_signature_dim: int = 836


class COMETActor(nn.Module):
    def __init__(self, model_config: ModelConfig, dims: ModelDimensions) -> None:
        super().__init__()
        self.dims = dims
        numeric_dim = dims.local_dim - 1
        self.zone_embedding = nn.Embedding(dims.cell_count + 1, model_config.zone_embedding_dim)
        self.local_encoder = _mlp(
            model_config.zone_embedding_dim + numeric_dim,
            model_config.hidden_dim,
            model_config.hidden_dim,
            model_config.dropout,
        )
        self.fleet_encoder = _mlp(
            dims.fleet_signature_dim,
            model_config.fleet_hidden_dim,
            model_config.hidden_dim,
            model_config.dropout,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 2, model_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_config.hidden_dim, dims.action_dim),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(model_config.hidden_dim, model_config.aux_hidden_dim),
            nn.ReLU(),
            nn.Linear(model_config.aux_hidden_dim, dims.cell_count),
        )

    def forward(
        self, local_obs: torch.Tensor, fleet_signature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zone_ids = local_obs[..., 0].long().clamp(min=0, max=self.dims.cell_count)
        numeric_features = local_obs[..., 1:]
        zone_embedding = self.zone_embedding(zone_ids)
        local_features = torch.cat([zone_embedding, numeric_features], dim=-1)
        local_hidden = self.local_encoder(local_features)
        fleet_hidden = self.fleet_encoder(fleet_signature)
        fleet_expanded = fleet_hidden.unsqueeze(1).expand(-1, local_hidden.size(1), -1)
        logits = self.policy_head(torch.cat([local_hidden, fleet_expanded], dim=-1))
        aux_prediction = self.aux_head(fleet_hidden)
        return logits, aux_prediction


class COMETCritic(nn.Module):
    def __init__(self, model_config: ModelConfig, dims: ModelDimensions) -> None:
        super().__init__()
        self.dims = dims
        numeric_dim = dims.local_dim - 1
        self.zone_embedding = nn.Embedding(dims.cell_count + 1, model_config.zone_embedding_dim)
        self.local_encoder = _mlp(
            model_config.zone_embedding_dim + numeric_dim,
            model_config.hidden_dim,
            model_config.hidden_dim,
            model_config.dropout,
        )
        self.fleet_encoder = _mlp(
            dims.fleet_signature_dim,
            model_config.fleet_hidden_dim,
            model_config.hidden_dim,
            model_config.dropout,
        )
        self.value_head = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 2, model_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_config.hidden_dim, 1),
        )

    def forward(
        self,
        local_obs: torch.Tensor,
        fleet_signature: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        zone_ids = local_obs[..., 0].long().clamp(min=0, max=self.dims.cell_count)
        numeric_features = local_obs[..., 1:]
        zone_embedding = self.zone_embedding(zone_ids)
        local_features = torch.cat([zone_embedding, numeric_features], dim=-1)
        local_hidden = self.local_encoder(local_features)
        mask = agent_mask.unsqueeze(-1)
        pooled_local = (local_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        fleet_hidden = self.fleet_encoder(fleet_signature)
        values = self.value_head(torch.cat([pooled_local, fleet_hidden], dim=-1))
        return values.squeeze(-1)


def infer_model_dimensions(cell_count: int) -> ModelDimensions:
    fleet_signature_dim = cell_count * 4 * 3 + cell_count + 4
    return ModelDimensions(cell_count=cell_count, fleet_signature_dim=fleet_signature_dim)
