from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import ModelConfig, SetEncoderConfig, TemporalConfig
from .constants import ACTION_DIM, AUXILIARY_TARGETS, COST_NAMES, LOCAL_OBS_DIM, MODE_COUNT


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = values * mask.unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return weighted.sum(dim=1) / denom


@dataclass(slots=True)
class ModelDimensions:
    local_dim: int = LOCAL_OBS_DIM
    action_dim: int = ACTION_DIM
    cell_count: int = 64
    fleet_signature_dim: int = 847
    temporal_feature_dim: int = 80
    charger_count: int = 5


@dataclass(slots=True)
class ActorOutput:
    logits: torch.Tensor
    aux_predictions: dict[str, torch.Tensor]
    vehicle_tokens: torch.Tensor
    fleet_context: torch.Tensor
    temporal_context: torch.Tensor
    uncertainty_proxy: torch.Tensor | None = None


@dataclass(slots=True)
class CriticOutput:
    mean: torch.Tensor
    variance: torch.Tensor
    ensemble_values: torch.Tensor


class RunningMeanStd(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("var", torch.ones(feature_dim))
        self.register_buffer("count", torch.tensor(eps))

    def update(self, values: torch.Tensor | np.ndarray) -> None:
        tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
        tensor = tensor.detach().float().reshape(-1, self.mean.shape[0])
        if tensor.numel() == 0:
            return
        batch_mean = tensor.mean(dim=0)
        batch_var = tensor.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(tensor.shape[0]), device=tensor.device)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = delta.pow(2) * self.count * batch_count / total
        new_var = (m_a + m_b + correction) / total.clamp_min(1e-6)
        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(1e-6))
        self.count.copy_(total)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / torch.sqrt(self.var + 1e-6)


class ObservationNormalizer(nn.Module):
    def __init__(self, dims: ModelDimensions) -> None:
        super().__init__()
        self.local_numeric = RunningMeanStd(dims.local_dim - 1)
        self.fleet_numeric = RunningMeanStd(dims.fleet_signature_dim)
        self.temporal_numeric = RunningMeanStd(dims.temporal_feature_dim)

    def update(self, observation: dict[str, torch.Tensor | np.ndarray]) -> None:
        local_obs = observation["local_obs"]
        fleet = observation["fleet_signature"]
        temporal = observation["temporal_history"]
        if isinstance(local_obs, np.ndarray):
            self.local_numeric.update(local_obs[..., 1:])
        else:
            self.local_numeric.update(local_obs[..., 1:].detach())
        self.fleet_numeric.update(fleet)
        self.temporal_numeric.update(temporal)

    def normalize(self, observation: dict[str, torch.Tensor], update_stats: bool = False) -> dict[str, torch.Tensor]:
        if update_stats:
            self.update(observation)
        normalized = dict(observation)
        zone_ids = observation["local_obs"][..., :1]
        local_numeric = self.local_numeric.normalize(observation["local_obs"][..., 1:])
        normalized["local_obs"] = torch.cat([zone_ids, local_numeric], dim=-1)
        normalized["fleet_signature"] = self.fleet_numeric.normalize(observation["fleet_signature"])
        normalized["temporal_history"] = self.temporal_numeric.normalize(observation["temporal_history"])
        return normalized


class VehicleTokenEncoder(nn.Module):
    def __init__(self, model_config: ModelConfig, dims: ModelDimensions) -> None:
        super().__init__()
        self.dims = dims
        self.zone_embedding = nn.Embedding(dims.cell_count + 1, model_config.zone_embedding_dim)
        numeric_dim = dims.local_dim - 1
        self.numeric_encoder = _mlp(
            numeric_dim,
            model_config.hidden_dim,
            model_config.vehicle_token_dim,
            model_config.dropout,
        )
        self.output = nn.Linear(
            model_config.vehicle_token_dim + model_config.zone_embedding_dim,
            model_config.vehicle_token_dim,
        )
        self.norm = nn.LayerNorm(model_config.vehicle_token_dim) if model_config.use_layer_norm else nn.Identity()

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        zone_ids = local_obs[..., 0].long().clamp(min=0, max=self.dims.cell_count)
        zone_embedding = self.zone_embedding(zone_ids)
        numeric_hidden = self.numeric_encoder(local_obs[..., 1:])
        tokens = self.output(torch.cat([zone_embedding, numeric_hidden], dim=-1))
        return self.norm(tokens)


class FleetSetEncoder(nn.Module):
    def __init__(
        self,
        set_config: SetEncoderConfig,
        model_config: ModelConfig,
        dims: ModelDimensions,
    ) -> None:
        super().__init__()
        token_dim = model_config.vehicle_token_dim
        self.set_type = set_config.type
        self.pooling = set_config.pooling
        self.use_skip = set_config.use_fleet_signature_skip
        self.phi = _mlp(token_dim, set_config.hidden_dim, token_dim, model_config.dropout)
        self.rho = _mlp(token_dim, set_config.hidden_dim, model_config.hidden_dim, model_config.dropout)
        self.fleet_skip = _mlp(
            dims.fleet_signature_dim,
            model_config.fleet_hidden_dim,
            model_config.hidden_dim,
            model_config.dropout,
        )
        if self.set_type == "set_transformer":
            self.inducing = nn.Parameter(
                torch.randn(1, set_config.num_inducing_points, token_dim) * 0.02
            )
            self.induced_attn = nn.MultiheadAttention(
                token_dim,
                num_heads=set_config.num_heads,
                batch_first=True,
            )
            self.token_attn = nn.MultiheadAttention(
                token_dim,
                num_heads=set_config.num_heads,
                batch_first=True,
            )
        else:
            self.inducing = None
            self.induced_attn = None
            self.token_attn = None

    def forward(
        self,
        vehicle_tokens: torch.Tensor,
        agent_mask: torch.Tensor,
        fleet_signature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = agent_mask <= 0
        if self.set_type == "set_transformer":
            assert self.inducing is not None
            assert self.induced_attn is not None
            assert self.token_attn is not None
            induced = self.inducing.expand(vehicle_tokens.shape[0], -1, -1)
            induced_hidden, _ = self.induced_attn(
                induced,
                vehicle_tokens,
                vehicle_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            encoded_tokens, _ = self.token_attn(
                vehicle_tokens,
                induced_hidden,
                induced_hidden,
                need_weights=False,
            )
            phi_tokens = self.phi(encoded_tokens)
        else:
            phi_tokens = self.phi(vehicle_tokens)

        masked_tokens = phi_tokens * agent_mask.unsqueeze(-1)
        pooled = masked_tokens.sum(dim=1)
        if self.pooling == "mean":
            pooled = pooled / agent_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        fleet_context = self.rho(pooled)
        if self.use_skip and fleet_signature is not None:
            fleet_context = fleet_context + self.fleet_skip(fleet_signature)
        return fleet_context


class TemporalEncoder(nn.Module):
    def __init__(self, config: TemporalConfig, model_config: ModelConfig, dims: ModelDimensions) -> None:
        super().__init__()
        self.encoder_type = config.encoder_type
        input_dim = dims.temporal_feature_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        if config.encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=4,
                dim_feedforward=max(config.hidden_dim * 2, 64),
                dropout=model_config.dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

    def forward(self, temporal_history: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(temporal_history)
        if self.encoder_type == "gru":
            _, hidden = self.encoder(projected)
            return hidden[-1]
        encoded = self.encoder(projected)
        return encoded[:, -1]


class ObservationBackboneV2(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        set_config: SetEncoderConfig,
        temporal_config: TemporalConfig,
        dims: ModelDimensions,
    ) -> None:
        super().__init__()
        self.vehicle_encoder = VehicleTokenEncoder(model_config, dims)
        self.set_encoder = FleetSetEncoder(set_config, model_config, dims)
        self.temporal_encoder = TemporalEncoder(temporal_config, model_config, dims)
        self.fleet_skip = _mlp(
            dims.fleet_signature_dim,
            model_config.fleet_hidden_dim,
            model_config.hidden_dim,
            model_config.dropout,
        )

    def forward(self, observation: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vehicle_tokens = self.vehicle_encoder(observation["local_obs"])
        fleet_context = self.set_encoder(
            vehicle_tokens,
            observation["agent_mask"],
            observation.get("fleet_signature"),
        )
        temporal_context = self.temporal_encoder(observation["temporal_history"])
        fleet_skip = self.fleet_skip(observation["fleet_signature"])
        return vehicle_tokens, fleet_context, temporal_context, fleet_skip


class COMETActorV2(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        set_config: SetEncoderConfig,
        temporal_config: TemporalConfig,
        dims: ModelDimensions,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.backbone = ObservationBackboneV2(model_config, set_config, temporal_config, dims)
        policy_input_dim = (
            model_config.vehicle_token_dim
            + model_config.hidden_dim
            + temporal_config.hidden_dim
            + model_config.hidden_dim
        )
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, model_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_config.hidden_dim, dims.action_dim),
        )
        self.aux_heads = nn.ModuleDict(
            {
                "next_demand": nn.Sequential(
                    nn.Linear(model_config.hidden_dim + temporal_config.hidden_dim, model_config.aux_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(model_config.aux_hidden_dim, dims.cell_count),
                ),
                "charger_occupancy": nn.Sequential(
                    nn.Linear(model_config.hidden_dim + temporal_config.hidden_dim, model_config.aux_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(model_config.aux_hidden_dim, dims.charger_count),
                ),
                "travel_time_residual": nn.Sequential(
                    nn.Linear(model_config.hidden_dim + temporal_config.hidden_dim, model_config.aux_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(model_config.aux_hidden_dim, 1),
                ),
            }
        )

    def forward(self, observation: dict[str, torch.Tensor]) -> ActorOutput:
        vehicle_tokens, fleet_context, temporal_context, fleet_skip = self.backbone(observation)
        shared_context = torch.cat([fleet_context, temporal_context, fleet_skip], dim=-1)
        repeated_context = shared_context.unsqueeze(1).expand(-1, vehicle_tokens.shape[1], -1)
        logits = self.policy_head(torch.cat([vehicle_tokens, repeated_context], dim=-1))
        aux_input = torch.cat([fleet_context, temporal_context], dim=-1)
        aux_predictions = {
            name: head(aux_input) for name, head in self.aux_heads.items()
        }
        return ActorOutput(
            logits=logits,
            aux_predictions=aux_predictions,
            vehicle_tokens=vehicle_tokens,
            fleet_context=fleet_context,
            temporal_context=temporal_context,
        )


class PermutationInvariantCentralizedCritic(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        set_config: SetEncoderConfig,
        temporal_config: TemporalConfig,
        dims: ModelDimensions,
    ) -> None:
        super().__init__()
        self.backbone = ObservationBackboneV2(model_config, set_config, temporal_config, dims)
        self.value_head = nn.Sequential(
            nn.Linear(model_config.hidden_dim + temporal_config.hidden_dim + model_config.hidden_dim, model_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_config.hidden_dim, 1),
        )

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        _, fleet_context, temporal_context, fleet_skip = self.backbone(observation)
        value = self.value_head(torch.cat([fleet_context, temporal_context, fleet_skip], dim=-1))
        return value.squeeze(-1)


class EnsembleCritic(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        set_config: SetEncoderConfig,
        temporal_config: TemporalConfig,
        dims: ModelDimensions,
        ensemble_size: int | None = None,
    ) -> None:
        super().__init__()
        size = ensemble_size or model_config.critic_ensemble_size
        self.members = nn.ModuleList(
            [
                PermutationInvariantCentralizedCritic(
                    model_config,
                    set_config,
                    temporal_config,
                    dims,
                )
                for _ in range(size)
            ]
        )

    def forward(self, observation: dict[str, torch.Tensor]) -> CriticOutput:
        ensemble_values = torch.stack([member(observation) for member in self.members], dim=-1)
        return CriticOutput(
            mean=ensemble_values.mean(dim=-1),
            variance=ensemble_values.var(dim=-1, unbiased=False),
            ensemble_values=ensemble_values,
        )


class CostCritic(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        set_config: SetEncoderConfig,
        temporal_config: TemporalConfig,
        dims: ModelDimensions,
    ) -> None:
        super().__init__()
        self.critics = nn.ModuleDict(
            {
                name: EnsembleCritic(model_config, set_config, temporal_config, dims)
                for name in COST_NAMES
            }
        )

    def forward(self, observation: dict[str, torch.Tensor]) -> dict[str, CriticOutput]:
        return {name: critic(observation) for name, critic in self.critics.items()}


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
        pooled_local = _masked_mean(local_hidden, agent_mask)
        fleet_hidden = self.fleet_encoder(fleet_signature)
        values = self.value_head(torch.cat([pooled_local, fleet_hidden], dim=-1))
        return values.squeeze(-1)


def infer_model_dimensions(
    cell_count: int,
    charger_count: int = 5,
    history_len: int = 6,
) -> ModelDimensions:
    fleet_signature_dim = cell_count * MODE_COUNT * 3 + cell_count + charger_count * 2 + 5
    temporal_feature_dim = cell_count + 1 + charger_count * 2 + 2
    return ModelDimensions(
        cell_count=cell_count,
        fleet_signature_dim=fleet_signature_dim,
        temporal_feature_dim=temporal_feature_dim,
        charger_count=charger_count,
    )


def ensure_tensor_observation(
    observation: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensor_observation: dict[str, torch.Tensor] = {}
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            tensor_observation[key] = value.to(device=device, dtype=torch.float32)
        elif isinstance(value, np.ndarray):
            tensor_observation[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor_observation["local_obs"].ndim == 2:
        for key in list(tensor_observation.keys()):
            tensor_observation[key] = tensor_observation[key].unsqueeze(0)
    return tensor_observation
