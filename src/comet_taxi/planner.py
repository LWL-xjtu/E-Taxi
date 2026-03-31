from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .baselines import GreedyDispatchPolicy
from .constants import ACTION_ACCEPT_ORDER, ACTION_GO_CHARGE, ACTION_STAY
from .models import COMETActorV2, CostCritic, EnsembleCritic, ObservationNormalizer, ensure_tensor_observation


@dataclass(slots=True)
class PlannerOutput:
    actions: np.ndarray
    log_probs: np.ndarray
    fallback_mask: np.ndarray
    uncertainty_mask: np.ndarray
    uncertainty_values: np.ndarray


class CandidatePlanner:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.fallback = GreedyDispatchPolicy(config)

    def _candidate_score(
        self,
        slot: int,
        action: int,
        observation: dict[str, np.ndarray],
        logits_row: np.ndarray,
        value_mean: float,
        cost_means: np.ndarray,
    ) -> float:
        demand_vector = observation.get("demand_vector")
        zone = int(observation["local_obs"][slot, 0]) - 1
        soc = float(observation["local_obs"][slot, 4])
        score = float(logits_row[action]) + 0.1 * value_mean
        if action == ACTION_ACCEPT_ORDER and demand_vector is not None and zone >= 0:
            score += 1.0 + float(demand_vector[zone])
        if action == ACTION_GO_CHARGE:
            score += 0.5 if soc < self.config.planner.risk_trigger_soc else -0.1
        if action == ACTION_STAY and demand_vector is not None and zone >= 0:
            score += 0.1 * float(demand_vector[zone])
        score -= 0.5 * float(cost_means.sum())
        return score

    def select_actions(
        self,
        actor: COMETActorV2,
        critic: EnsembleCritic,
        cost_critic: CostCritic,
        observation: dict[str, np.ndarray],
        device: torch.device,
        env: Any,
        normalizer: ObservationNormalizer | None = None,
        planner_enabled: bool = True,
    ) -> PlannerOutput:
        actions = np.zeros(self.config.env.nmax, dtype=np.int64)
        log_probs = np.zeros(self.config.env.nmax, dtype=np.float32)
        fallback_mask = np.zeros(self.config.env.nmax, dtype=np.float32)
        uncertainty_mask = np.zeros(self.config.env.nmax, dtype=np.float32)
        fallback_actions = self.fallback.act(observation)

        tensor_observation = ensure_tensor_observation(observation, device)
        if normalizer is not None:
            tensor_observation = normalizer.normalize(tensor_observation, update_stats=False)

        with torch.no_grad():
            actor_output = actor(tensor_observation)
            critic_output = critic(tensor_observation)
            cost_outputs = cost_critic(tensor_observation)

        logits = actor_output.logits.squeeze(0).cpu().numpy()
        value_mean = float(critic_output.mean.squeeze(0).item())
        uncertainty_value = float(torch.sqrt(critic_output.variance.squeeze(0) + 1e-8).item())
        cost_means = np.asarray(
            [float(output.mean.squeeze(0).item()) for output in cost_outputs.values()],
            dtype=np.float32,
        )
        action_mask = observation["action_mask"]
        candidates = env.generate_candidate_actions(observation, self.config.planner.top_k_zones)

        for slot in range(self.config.env.nmax):
            if observation["agent_mask"][slot] <= 0:
                continue
            masked_logits = np.where(action_mask[slot] > 0, logits[slot], -1e9)
            if not planner_enabled or self.config.planner.planner_mode == "policy":
                chosen = int(masked_logits.argmax())
            else:
                soc = float(observation["local_obs"][slot, 4])
                charger_queue = float(np.max(observation.get("charger_queue", np.zeros(1, dtype=np.float32))))
                force_fallback = soc < self.config.planner.risk_trigger_soc or charger_queue > self.config.env.max_queue_length / 2
                if uncertainty_value > self.config.online_finetune.uncertainty_threshold:
                    uncertainty_mask[slot] = 1.0
                    force_fallback = True
                if force_fallback:
                    chosen = int(fallback_actions[slot])
                    fallback_mask[slot] = 1.0
                else:
                    candidate_scores = [
                        (
                            self._candidate_score(
                                slot,
                                action,
                                observation,
                                masked_logits,
                                value_mean,
                                cost_means,
                            ),
                            action,
                        )
                        for action in candidates[slot]
                    ]
                    chosen = int(max(candidate_scores, key=lambda item: item[0])[1])
            actions[slot] = chosen
            probabilities = torch.softmax(torch.as_tensor(masked_logits), dim=-1)
            log_probs[slot] = float(torch.log(probabilities[chosen].clamp_min(1e-8)).item())

        uncertainty_values = np.full(self.config.env.nmax, uncertainty_value, dtype=np.float32)
        return PlannerOutput(
            actions=actions,
            log_probs=log_probs,
            fallback_mask=fallback_mask,
            uncertainty_mask=uncertainty_mask,
            uncertainty_values=uncertainty_values,
        )
