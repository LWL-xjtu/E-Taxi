from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class RolloutBatch:
    local_obs: torch.Tensor
    fleet_signature: torch.Tensor
    agent_mask: torch.Tensor
    action_mask: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    aux_targets: torch.Tensor


class RolloutBuffer:
    def __init__(self) -> None:
        self.observations: list[dict[str, np.ndarray]] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[np.ndarray] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.aux_targets: list[np.ndarray] = []

    def add(
        self,
        observation: dict[str, np.ndarray],
        actions: np.ndarray,
        log_probs: np.ndarray,
        value: float,
        reward: float,
        done: bool,
        aux_target: np.ndarray,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(actions.astype(np.int64, copy=False))
        self.log_probs.append(log_probs.astype(np.float32, copy=False))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.aux_targets.append(aux_target.astype(np.float32, copy=False))

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
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[step]
            delta = rewards[step] + gamma * values[step + 1] * non_terminal - values[step]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[step] = gae
        returns = advantages + values[:-1]

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
            returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
            advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
            aux_targets=torch.as_tensor(
                np.stack(self.aux_targets), dtype=torch.float32, device=device
            ),
        )
