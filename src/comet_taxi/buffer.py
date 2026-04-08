from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .constants import COST_NAMES


@dataclass(slots=True)
class RolloutBatch:
    local_obs: torch.Tensor
    fleet_signature: torch.Tensor
    temporal_history: torch.Tensor
    agent_mask: torch.Tensor
    action_mask: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    costs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    cost_returns: torch.Tensor
    cost_advantages: torch.Tensor
    aux_next_demand: torch.Tensor
    aux_charger_occupancy: torch.Tensor
    aux_travel_time_residual: torch.Tensor
    uncertainty: torch.Tensor


@dataclass(slots=True)
class ReplayBatch:
    local_obs: torch.Tensor
    fleet_signature: torch.Tensor
    temporal_history: torch.Tensor
    agent_mask: torch.Tensor
    action_mask: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    returns: torch.Tensor
    costs: torch.Tensor
    next_local_obs: torch.Tensor
    next_fleet_signature: torch.Tensor
    next_temporal_history: torch.Tensor
    next_agent_mask: torch.Tensor
    next_action_mask: torch.Tensor
    dones: torch.Tensor
    aux_next_demand: torch.Tensor
    aux_charger_occupancy: torch.Tensor
    aux_travel_time_residual: torch.Tensor


class RolloutBuffer:
    def __init__(self) -> None:
        self.observations: list[dict[str, np.ndarray]] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[np.ndarray] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.costs: list[np.ndarray] = []
        self.aux_targets: list[dict[str, np.ndarray | float]] = []
        self.uncertainty: list[float] = []

    def add(
        self,
        observation: dict[str, np.ndarray],
        actions: np.ndarray,
        log_probs: np.ndarray,
        value: float,
        reward: float,
        done: bool,
        costs: dict[str, float],
        aux_targets: dict[str, np.ndarray | float],
        uncertainty: float,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(actions.astype(np.int64, copy=False))
        self.log_probs.append(log_probs.astype(np.float32, copy=False))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.costs.append(
            np.asarray([costs[name] for name in COST_NAMES], dtype=np.float32)
        )
        self.aux_targets.append(aux_targets)
        self.uncertainty.append(float(uncertainty))

    def compute_batch(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> RolloutBatch:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values + [last_value], dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        cost_matrix = np.stack(self.costs).astype(np.float32)
        advantages = np.zeros_like(rewards)
        cost_advantages = np.zeros_like(cost_matrix)
        gae = 0.0
        cost_gae = np.zeros(cost_matrix.shape[1], dtype=np.float32)
        for step in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[step]
            delta = rewards[step] + gamma * values[step + 1] * non_terminal - values[step]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[step] = gae
            cost_gae = cost_matrix[step] + gamma * gae_lambda * non_terminal * cost_gae
            cost_advantages[step] = cost_gae
        returns = advantages + values[:-1]
        cost_returns = cost_advantages.copy()

        return RolloutBatch(
            local_obs=torch.as_tensor(
                np.stack([item["local_obs"] for item in self.observations]),
                dtype=torch.float32,
                device=device,
            ),
            fleet_signature=torch.as_tensor(
                np.stack([item["fleet_signature"] for item in self.observations]),
                dtype=torch.float32,
                device=device,
            ),
            temporal_history=torch.as_tensor(
                np.stack([item["temporal_history"] for item in self.observations]),
                dtype=torch.float32,
                device=device,
            ),
            agent_mask=torch.as_tensor(
                np.stack([item["agent_mask"] for item in self.observations]),
                dtype=torch.float32,
                device=device,
            ),
            action_mask=torch.as_tensor(
                np.stack([item["action_mask"] for item in self.observations]),
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.as_tensor(np.stack(self.actions), dtype=torch.long, device=device),
            old_log_probs=torch.as_tensor(
                np.stack(self.log_probs), dtype=torch.float32, device=device
            ),
            costs=torch.as_tensor(cost_matrix, dtype=torch.float32, device=device),
            returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
            advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
            cost_returns=torch.as_tensor(cost_returns, dtype=torch.float32, device=device),
            cost_advantages=torch.as_tensor(
                cost_advantages, dtype=torch.float32, device=device
            ),
            aux_next_demand=torch.as_tensor(
                np.stack([item["next_demand"] for item in self.aux_targets]),
                dtype=torch.float32,
                device=device,
            ),
            aux_charger_occupancy=torch.as_tensor(
                np.stack([item["charger_occupancy"] for item in self.aux_targets]),
                dtype=torch.float32,
                device=device,
            ),
            aux_travel_time_residual=torch.as_tensor(
                np.asarray(
                    [[float(item["travel_time_residual"])] for item in self.aux_targets],
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=device,
            ),
            uncertainty=torch.as_tensor(
                np.asarray(self.uncertainty, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
        )


class TransitionReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.storage: list[dict[str, np.ndarray | float]] = []

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: dict[str, np.ndarray | float]) -> None:
        if len(self.storage) >= self.capacity:
            self.storage.pop(0)
        self.storage.append(transition)

    def extend_from_rollout(
        self,
        observations: list[dict[str, np.ndarray]],
        actions: list[np.ndarray],
        rewards: list[float],
        costs: list[np.ndarray],
        dones: list[bool],
        aux_targets: list[dict[str, np.ndarray | float]],
    ) -> None:
        for index in range(len(observations) - 1):
            self.add(
                {
                    "local_obs": observations[index]["local_obs"],
                    "fleet_signature": observations[index]["fleet_signature"],
                    "temporal_history": observations[index]["temporal_history"],
                    "agent_mask": observations[index]["agent_mask"],
                    "action_mask": observations[index]["action_mask"],
                    "actions": actions[index],
                    "rewards": rewards[index],
                    "returns": rewards[index],
                    "costs": costs[index],
                    "next_local_obs": observations[index + 1]["local_obs"],
                    "next_fleet_signature": observations[index + 1]["fleet_signature"],
                    "next_temporal_history": observations[index + 1]["temporal_history"],
                    "next_agent_mask": observations[index + 1]["agent_mask"],
                    "next_action_mask": observations[index + 1]["action_mask"],
                    "dones": float(dones[index]),
                    "aux_next_demand": aux_targets[index]["next_demand"],
                    "aux_charger_occupancy": aux_targets[index]["charger_occupancy"],
                    "aux_travel_time_residual": np.asarray(
                        [float(aux_targets[index]["travel_time_residual"])], dtype=np.float32
                    ),
                }
            )

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        indices = np.random.choice(len(self.storage), size=min(batch_size, len(self.storage)), replace=False)
        batch = [self.storage[index] for index in indices]
        return ReplayBatch(
            local_obs=torch.as_tensor(np.stack([item["local_obs"] for item in batch]), dtype=torch.float32, device=device),
            fleet_signature=torch.as_tensor(np.stack([item["fleet_signature"] for item in batch]), dtype=torch.float32, device=device),
            temporal_history=torch.as_tensor(np.stack([item["temporal_history"] for item in batch]), dtype=torch.float32, device=device),
            agent_mask=torch.as_tensor(np.stack([item["agent_mask"] for item in batch]), dtype=torch.float32, device=device),
            action_mask=torch.as_tensor(np.stack([item["action_mask"] for item in batch]), dtype=torch.float32, device=device),
            actions=torch.as_tensor(np.stack([item["actions"] for item in batch]), dtype=torch.long, device=device),
            rewards=torch.as_tensor(np.asarray([item["rewards"] for item in batch], dtype=np.float32), dtype=torch.float32, device=device),
            returns=torch.as_tensor(np.asarray([item["returns"] for item in batch], dtype=np.float32), dtype=torch.float32, device=device),
            costs=torch.as_tensor(np.stack([item["costs"] for item in batch]), dtype=torch.float32, device=device),
            next_local_obs=torch.as_tensor(np.stack([item["next_local_obs"] for item in batch]), dtype=torch.float32, device=device),
            next_fleet_signature=torch.as_tensor(np.stack([item["next_fleet_signature"] for item in batch]), dtype=torch.float32, device=device),
            next_temporal_history=torch.as_tensor(np.stack([item["next_temporal_history"] for item in batch]), dtype=torch.float32, device=device),
            next_agent_mask=torch.as_tensor(np.stack([item["next_agent_mask"] for item in batch]), dtype=torch.float32, device=device),
            next_action_mask=torch.as_tensor(np.stack([item["next_action_mask"] for item in batch]), dtype=torch.float32, device=device),
            dones=torch.as_tensor(np.asarray([item["dones"] for item in batch], dtype=np.float32), dtype=torch.float32, device=device),
            aux_next_demand=torch.as_tensor(np.stack([item["aux_next_demand"] for item in batch]), dtype=torch.float32, device=device),
            aux_charger_occupancy=torch.as_tensor(np.stack([item["aux_charger_occupancy"] for item in batch]), dtype=torch.float32, device=device),
            aux_travel_time_residual=torch.as_tensor(np.stack([item["aux_travel_time_residual"] for item in batch]), dtype=torch.float32, device=device),
        )


def replay_from_arrays(arrays: dict[str, np.ndarray], device: torch.device) -> ReplayBatch:
    return ReplayBatch(
        local_obs=torch.as_tensor(arrays["local_obs"], dtype=torch.float32, device=device),
        fleet_signature=torch.as_tensor(arrays["fleet_signature"], dtype=torch.float32, device=device),
        temporal_history=torch.as_tensor(arrays["temporal_history"], dtype=torch.float32, device=device),
        agent_mask=torch.as_tensor(arrays["agent_mask"], dtype=torch.float32, device=device),
        action_mask=torch.as_tensor(arrays["action_mask"], dtype=torch.float32, device=device),
        actions=torch.as_tensor(arrays["actions"], dtype=torch.long, device=device),
        rewards=torch.as_tensor(arrays["rewards"], dtype=torch.float32, device=device),
        returns=torch.as_tensor(arrays["returns"], dtype=torch.float32, device=device),
        costs=torch.as_tensor(arrays["costs"], dtype=torch.float32, device=device),
        next_local_obs=torch.as_tensor(arrays["next_local_obs"], dtype=torch.float32, device=device),
        next_fleet_signature=torch.as_tensor(arrays["next_fleet_signature"], dtype=torch.float32, device=device),
        next_temporal_history=torch.as_tensor(arrays["next_temporal_history"], dtype=torch.float32, device=device),
        next_agent_mask=torch.as_tensor(arrays["next_agent_mask"], dtype=torch.float32, device=device),
        next_action_mask=torch.as_tensor(arrays["next_action_mask"], dtype=torch.float32, device=device),
        dones=torch.as_tensor(arrays["dones"], dtype=torch.float32, device=device),
        aux_next_demand=torch.as_tensor(arrays["aux_next_demand"], dtype=torch.float32, device=device),
        aux_charger_occupancy=torch.as_tensor(arrays["aux_charger_occupancy"], dtype=torch.float32, device=device),
        aux_travel_time_residual=torch.as_tensor(arrays["aux_travel_time_residual"], dtype=torch.float32, device=device),
    )
