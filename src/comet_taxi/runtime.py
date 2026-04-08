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


class UncertaintyCalibrator:
    def __init__(self, eps: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = float(eps)

    def update(self, value: float) -> None:
        value = float(value)
        delta = value - self.mean
        total = self.count + 1.0
        new_mean = self.mean + delta / total
        m_a = self.var * self.count
        m_b = 0.0
        correction = delta * delta * self.count / total
        self.mean = new_mean
        self.var = max((m_a + m_b + correction) / max(total, 1e-6), 1e-6)
        self.count = total

    def z_score(self, value: float) -> float:
        return float((float(value) - self.mean) / max(self.var ** 0.5, 1e-6))

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict: dict[str, float] | None) -> None:
        if not state_dict:
            return
        self.mean = float(state_dict.get("mean", self.mean))
        self.var = max(float(state_dict.get("var", self.var)), 1e-6)
        self.count = max(float(state_dict.get("count", self.count)), 1e-4)


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


def _select_policy_actions(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    sample_actions: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    masked = masked_logits(logits, action_mask)
    actions = torch.zeros(agent_mask.shape[0], dtype=torch.long, device=logits.device)
    log_probs = torch.zeros(agent_mask.shape[0], dtype=torch.float32, device=logits.device)
    policy_selected_mask = torch.zeros(agent_mask.shape[0], dtype=torch.float32, device=logits.device)
    for slot in range(agent_mask.shape[0]):
        if agent_mask[slot] <= 0:
            continue
        distribution = Categorical(logits=masked[slot])
        selected_action = distribution.sample() if sample_actions else torch.argmax(masked[slot])
        actions[slot] = selected_action
        log_probs[slot] = distribution.log_prob(selected_action)
        policy_selected_mask[slot] = 1.0
    zero_mask = np.zeros_like(agent_mask.detach().cpu().numpy())
    return actions.cpu().numpy(), log_probs.cpu().numpy(), {
        "policy_selected_mask": policy_selected_mask.cpu().numpy(),
        "planner_selected_mask": zero_mask,
        "fallback_mask": zero_mask,
        "uncertainty_mask": zero_mask,
        "uncertainty_values": zero_mask,
        "uncertainty_z_values": zero_mask,
        "uncertainty": 0.0,
    }


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
    execution_mode: str = "planner",
    uncertainty_calibrator: UncertaintyCalibrator | None = None,
    current_episode: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    if isinstance(actor, COMETActorV2):
        tensor_observation = ensure_tensor_observation(observation, device)
        if normalizer is not None:
            tensor_observation = normalizer.normalize(tensor_observation, update_stats=False)
        with torch.no_grad():
            actor_output = actor(tensor_observation)
        logits = actor_output.logits.squeeze(0)
        action_mask = torch.as_tensor(observation["action_mask"], dtype=torch.float32, device=device)
        agent_mask = torch.as_tensor(observation["agent_mask"], dtype=torch.float32, device=device)
        if execution_mode == "policy_sample":
            return _select_policy_actions(logits, action_mask, agent_mask, sample_actions=True)
        if execution_mode == "policy_greedy" or planner is None or critic is None or cost_critic is None or env is None:
            return _select_policy_actions(logits, action_mask, agent_mask, sample_actions=False)

        planner_output = planner.select_actions(
            actor=actor,
            critic=critic,
            cost_critic=cost_critic,
            observation=observation,
            device=device,
            env=env,
            normalizer=normalizer,
            planner_enabled=planner_enabled,
            uncertainty_calibrator=uncertainty_calibrator,
            current_episode=current_episode,
        )
        return planner_output.actions, planner_output.log_probs, {
            "policy_selected_mask": planner_output.policy_selected_mask,
            "planner_selected_mask": planner_output.planner_selected_mask,
            "fallback_mask": planner_output.fallback_mask,
            "uncertainty_mask": planner_output.uncertainty_mask,
            "uncertainty_values": planner_output.uncertainty_values,
            "uncertainty_z_values": planner_output.uncertainty_z_values,
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
    sample_actions = execution_mode == "policy_sample"
    return _select_policy_actions(logits, action_mask, agent_mask, sample_actions=sample_actions)


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
