from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from .models import (
    COMETActor,
    COMETActorV2,
    CostCritic,
    EnsembleCritic,
    ObservationNormalizer,
    ensure_tensor_observation,
)
from .planner import CandidatePlanner


def resolve_device(device_name: str) -> torch.device:
    lowered = device_name.lower()
    if lowered == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if lowered.startswith("cuda") and torch.cuda.is_available():
        return torch.device(lowered)
    return torch.device("cpu")


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(action_mask <= 0, -1e9)
    empty_rows = action_mask.sum(dim=-1, keepdim=True) <= 0
    return torch.where(empty_rows, torch.zeros_like(masked), masked)


def action_selection_from_policy(
    actor: COMETActor | COMETActorV2,
    observation: dict[str, Any],
    device: torch.device,
    critic: EnsembleCritic | None = None,
    cost_critic: CostCritic | None = None,
    planner: CandidatePlanner | None = None,
    env: Any | None = None,
    normalizer: ObservationNormalizer | None = None,
    planner_enabled: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    if isinstance(actor, COMETActorV2):
        if planner is None or critic is None or cost_critic is None or env is None:
            tensor_observation = ensure_tensor_observation(observation, device)
            if normalizer is not None:
                tensor_observation = normalizer.normalize(tensor_observation, update_stats=False)
            with torch.no_grad():
                actor_output = actor(tensor_observation)
            logits = actor_output.logits.squeeze(0)
            action_mask = torch.as_tensor(observation["action_mask"], dtype=torch.float32, device=device)
            agent_mask = torch.as_tensor(observation["agent_mask"], dtype=torch.float32, device=device)
            masked = masked_logits(logits, action_mask)
            actions = torch.zeros(agent_mask.shape[0], dtype=torch.long, device=device)
            log_probs = torch.zeros(agent_mask.shape[0], dtype=torch.float32, device=device)
            for slot in range(agent_mask.shape[0]):
                if agent_mask[slot] <= 0:
                    continue
                distribution = Categorical(logits=masked[slot])
                sampled_action = distribution.sample()
                actions[slot] = sampled_action
                log_probs[slot] = distribution.log_prob(sampled_action)
            return actions.cpu().numpy(), log_probs.cpu().numpy(), {
                "fallback_mask": np.zeros_like(observation["agent_mask"]),
                "uncertainty_mask": np.zeros_like(observation["agent_mask"]),
                "uncertainty": 0.0,
            }
        planner_output = planner.select_actions(
            actor=actor,
            critic=critic,
            cost_critic=cost_critic,
            observation=observation,
            device=device,
            env=env,
            normalizer=normalizer,
            planner_enabled=planner_enabled,
        )
        return planner_output.actions, planner_output.log_probs, {
            "fallback_mask": planner_output.fallback_mask,
            "uncertainty_mask": planner_output.uncertainty_mask,
            "uncertainty": float(planner_output.uncertainty_values.max(initial=0.0)),
        }

    local_obs = torch.as_tensor(observation["local_obs"], dtype=torch.float32, device=device).unsqueeze(0)
    fleet_signature = torch.as_tensor(
        observation["fleet_signature"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    action_mask = torch.as_tensor(observation["action_mask"], dtype=torch.float32, device=device)
    agent_mask = torch.as_tensor(observation["agent_mask"], dtype=torch.float32, device=device)

    with torch.no_grad():
        logits, _ = actor(local_obs, fleet_signature)
    logits = logits.squeeze(0)
    masked = masked_logits(logits, action_mask)
    actions = torch.zeros(agent_mask.shape[0], dtype=torch.long, device=device)
    log_probs = torch.zeros(agent_mask.shape[0], dtype=torch.float32, device=device)
    for slot in range(agent_mask.shape[0]):
        if agent_mask[slot] <= 0:
            continue
        distribution = Categorical(logits=masked[slot])
        sampled_action = distribution.sample()
        actions[slot] = sampled_action
        log_probs[slot] = distribution.log_prob(sampled_action)
    return actions.cpu().numpy(), log_probs.cpu().numpy(), {
        "fallback_mask": np.zeros_like(observation["agent_mask"]),
        "uncertainty_mask": np.zeros_like(observation["agent_mask"]),
        "uncertainty": 0.0,
    }


def policy_statistics(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    actions: torch.Tensor,
    agent_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distribution = Categorical(logits=masked_logits(logits, action_mask))
    log_probs = distribution.log_prob(actions) * agent_mask
    entropy = distribution.entropy() * agent_mask
    return log_probs, entropy
