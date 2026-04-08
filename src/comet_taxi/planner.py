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
    policy_selected_mask: np.ndarray
    planner_selected_mask: np.ndarray
    fallback_mask: np.ndarray
    uncertainty_mask: np.ndarray
    uncertainty_values: np.ndarray
    uncertainty_z_values: np.ndarray


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
        uncertainty_value: float,
    ) -> float:
        demand_vector = observation.get("demand_vector")
        zone = int(observation["local_obs"][slot, 0]) - 1
        soc = float(observation["local_obs"][slot, 4])
        score = self.config.planner.actor_prior_weight * float(logits_row[action])
        score += self.config.planner.value_weight * float(value_mean)
        if action == ACTION_ACCEPT_ORDER and demand_vector is not None and zone >= 0:
            score += 1.0 + float(demand_vector[zone])
        if action == ACTION_GO_CHARGE:
            score += 0.5 if soc < self.config.planner.risk_trigger_soc else -0.1
        if action == ACTION_STAY and demand_vector is not None and zone >= 0:
            score += 0.1 * float(demand_vector[zone])
        score -= self.config.planner.cost_penalty_weight * float(cost_means.sum())
        score -= self.config.planner.uncertainty_penalty_weight * float(uncertainty_value)
        return score

    def _row_entropy(self, masked_logits: np.ndarray) -> float:
        tensor = torch.as_tensor(masked_logits, dtype=torch.float32)
        probabilities = torch.softmax(tensor, dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum()
        return float(entropy.item())

    def _row_margin(self, masked_logits: np.ndarray) -> float:
        valid = masked_logits[np.isfinite(masked_logits) & (masked_logits > -1e8)]
        if valid.size <= 1:
            return float("inf")
        top_two = np.sort(valid)[-2:]
        return float(top_two[-1] - top_two[-2])

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
        uncertainty_calibrator: Any | None = None,
        current_episode: int = 0,
    ) -> PlannerOutput:
        actions = np.zeros(self.config.env.nmax, dtype=np.int64)
        log_probs = np.zeros(self.config.env.nmax, dtype=np.float32)
        policy_selected_mask = np.zeros(self.config.env.nmax, dtype=np.float32)
        planner_selected_mask = np.zeros(self.config.env.nmax, dtype=np.float32)
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
        uncertainty_z = (
            uncertainty_calibrator.z_score(uncertainty_value)
            if uncertainty_calibrator is not None
            else uncertainty_value
        )
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
                policy_selected_mask[slot] = 1.0
            else:
                soc = float(observation["local_obs"][slot, 4])
                charger_queue = float(np.max(observation.get("charger_queue", np.zeros(1, dtype=np.float32))))
                actor_entropy = self._row_entropy(masked_logits)
                action_margin = self._row_margin(masked_logits)
                local_risk = (
                    soc < self.config.planner.risk_trigger_soc
                    or charger_queue > self.config.env.max_queue_length / 2
                    or actor_entropy > self.config.planner.actor_entropy_threshold
                    or action_margin < self.config.planner.action_margin_threshold
                )
                uncertainty_gate_active = current_episode > self.config.planner.uncertainty_warmup_episodes
                force_fallback = soc < self.config.planner.risk_trigger_soc or charger_queue > self.config.env.max_queue_length / 2
                if uncertainty_gate_active and local_risk and uncertainty_z > self.config.planner.uncertainty_z_threshold:
                    uncertainty_mask[slot] = 1.0
                    force_fallback = force_fallback or self.config.planner.enable_fallback
                if force_fallback and self.config.planner.enable_fallback:
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
                                uncertainty_value,
                            ),
                            action,
                        )
                        for action in candidates[slot]
                    ]
                    chosen = int(max(candidate_scores, key=lambda item: item[0])[1])
                    planner_selected_mask[slot] = 1.0
            actions[slot] = chosen
            probabilities = torch.softmax(torch.as_tensor(masked_logits), dim=-1)
            log_probs[slot] = float(torch.log(probabilities[chosen].clamp_min(1e-8)).item())

        uncertainty_values = np.full(self.config.env.nmax, uncertainty_value, dtype=np.float32)
        uncertainty_z_values = np.full(self.config.env.nmax, uncertainty_z, dtype=np.float32)
        return PlannerOutput(
            actions=actions,
            log_probs=log_probs,
            policy_selected_mask=policy_selected_mask,
            planner_selected_mask=planner_selected_mask,
            fallback_mask=fallback_mask,
            uncertainty_mask=uncertainty_mask,
            uncertainty_values=uncertainty_values,
            uncertainty_z_values=uncertainty_z_values,
        )
