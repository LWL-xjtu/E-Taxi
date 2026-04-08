
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .buffer import ReplayBatch, RolloutBatch, RolloutBuffer, TransitionReplayBuffer, replay_from_arrays
from .config import ExperimentConfig
from .constants import COST_NAMES
from .data import OfflineTransitionDataset, PreparedDataset, export_offline_transition_dataset
from .env import CometTaxiEnv
from .evaluation import evaluate_policy, write_reward_curve
from .models import (
    COMETActor,
    COMETActorV2,
    COMETCritic,
    CostCritic,
    EnsembleCritic,
    ObservationNormalizer,
    ensure_tensor_observation,
    infer_model_dimensions,
)
from .planner import CandidatePlanner
from .runtime import (
    UncertaintyCalibrator,
    action_selection_from_policy,
    policy_statistics,
    resolve_device,
)
from .utils import ensure_dir, seed_everything


class CometTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        dataset_root: str | Path,
        output_dir: str | Path,
        resume_checkpoint: str | Path | None = None,
    ) -> None:
        self.config = config
        self.dataset = PreparedDataset.load(dataset_root)
        self.env = CometTaxiEnv(self.dataset, config, seed=config.train.seed)
        self.device = resolve_device(config.train.device)
        self.dims = infer_model_dimensions(
            self.dataset.metadata["data"]["cell_count"],
            charger_count=len(self.dataset.metadata["charge_stations"]),
            history_len=config.temporal.history_len,
        )
        self.legacy_mode = config.model.variant == "legacy"
        if self.legacy_mode:
            self.actor: COMETActor | COMETActorV2 = COMETActor(config.model, self.dims).to(self.device)
            self.critic: COMETCritic | EnsembleCritic = COMETCritic(config.model, self.dims).to(self.device)
            self.cost_critic: CostCritic | None = None
            self.normalizer: ObservationNormalizer | None = None
        else:
            self.actor = COMETActorV2(config.model, config.set_encoder, config.temporal, self.dims).to(self.device)
            self.critic = EnsembleCritic(config.model, config.set_encoder, config.temporal, self.dims).to(self.device)
            self.cost_critic = CostCritic(config.model, config.set_encoder, config.temporal, self.dims).to(self.device)
            self.normalizer = ObservationNormalizer(self.dims).to(self.device)
        parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.cost_critic is not None:
            parameters += list(self.cost_critic.parameters())
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.output_dir = ensure_dir(output_dir)
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.history: list[dict[str, float | int]] = []
        self.offline_dataset_path = self.output_dir / "offline_dataset.npz"
        self.offline_dataset: OfflineTransitionDataset | None = None
        self.offline_replay: ReplayBatch | None = None
        self.online_replay = TransitionReplayBuffer(config.online_finetune.online_replay_capacity)
        self.planner = CandidatePlanner(config)
        self.constraint_multipliers = {name: config.safety.multiplier_init for name in COST_NAMES}
        self.constraint_ema = {name: 0.0 for name in COST_NAMES}
        self.scheduler_state = {"step": 0}
        self.start_episode = 1
        self.best_val_mean_team_reward = float("-inf")
        self.best_runtime_mean_team_reward = float("-inf")
        self.resume_wall_clock_seconds = 0.0
        self.uncertainty_calibrator = UncertaintyCalibrator()
        seed_everything(config.train.seed)
        if resume_checkpoint is not None:
            self.load_checkpoint(resume_checkpoint)

    def _slice_replay_batch(self, batch: ReplayBatch, indices: np.ndarray) -> ReplayBatch:
        return ReplayBatch(
            local_obs=batch.local_obs[indices],
            fleet_signature=batch.fleet_signature[indices],
            temporal_history=batch.temporal_history[indices],
            agent_mask=batch.agent_mask[indices],
            action_mask=batch.action_mask[indices],
            actions=batch.actions[indices],
            rewards=batch.rewards[indices],
            returns=batch.returns[indices],
            costs=batch.costs[indices],
            next_local_obs=batch.next_local_obs[indices],
            next_fleet_signature=batch.next_fleet_signature[indices],
            next_temporal_history=batch.next_temporal_history[indices],
            next_agent_mask=batch.next_agent_mask[indices],
            next_action_mask=batch.next_action_mask[indices],
            dones=batch.dones[indices],
            aux_next_demand=batch.aux_next_demand[indices],
            aux_charger_occupancy=batch.aux_charger_occupancy[indices],
            aux_travel_time_residual=batch.aux_travel_time_residual[indices],
        )

    def _save_checkpoint(self, episode: int) -> Path:
        checkpoint_path = self.checkpoint_dir / f"episode_{episode:04d}.pt"
        checkpoint = {
            "episode": episode,
            "config": self.config.to_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "cost_critic_state_dict": self.cost_critic.state_dict() if self.cost_critic is not None else {},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "normalizer_state_dict": self.normalizer.state_dict() if self.normalizer is not None else None,
            "constraint_multipliers": self.constraint_multipliers,
            "constraint_ema": self.constraint_ema,
            "scheduler_state": self.scheduler_state,
            "best_val_mean_team_reward": self.best_val_mean_team_reward,
            "best_runtime_mean_team_reward": self.best_runtime_mean_team_reward,
            "resume_wall_clock_seconds": self.resume_wall_clock_seconds,
            "uncertainty_calibrator_state_dict": self.uncertainty_calibrator.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")
        return checkpoint_path

    def _save_named_checkpoint(self, episode: int, target_name: str) -> None:
        checkpoint_path = self.checkpoint_dir / f"episode_{episode:04d}.pt"
        best_path = self.checkpoint_dir / target_name
        if checkpoint_path.exists():
            shutil.copy2(checkpoint_path, best_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if self.cost_critic is not None and checkpoint.get("cost_critic_state_dict"):
            self.cost_critic.load_state_dict(checkpoint["cost_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.normalizer is not None and checkpoint.get("normalizer_state_dict"):
            self.normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
            self.normalizer.to(self.device)
        self.constraint_multipliers.update(checkpoint.get("constraint_multipliers", {}))
        self.constraint_ema.update(checkpoint.get("constraint_ema", {}))
        self.scheduler_state.update(checkpoint.get("scheduler_state", {}))
        self.best_val_mean_team_reward = float(
            checkpoint.get("best_val_mean_team_reward", self.best_val_mean_team_reward)
        )
        self.best_runtime_mean_team_reward = float(
            checkpoint.get("best_runtime_mean_team_reward", self.best_runtime_mean_team_reward)
        )
        self.resume_wall_clock_seconds = float(
            checkpoint.get("resume_wall_clock_seconds", self.resume_wall_clock_seconds)
        )
        self.uncertainty_calibrator.load_state_dict(
            checkpoint.get("uncertainty_calibrator_state_dict")
        )
        self.start_episode = int(checkpoint.get("episode", 0)) + 1
        metrics_path = self.output_dir / "metrics.csv"
        if metrics_path.exists():
            existing = pd.read_csv(metrics_path)
            self.history = existing.to_dict("records")

    def _offline_ratio(self) -> float:
        decay_steps = max(self.config.online_finetune.offline_ratio_decay_steps, 1)
        progress = min(self.scheduler_state["step"] / decay_steps, 1.0)
        start = self.config.online_finetune.offline_ratio_start
        end = self.config.online_finetune.offline_ratio_end
        return float(start + (end - start) * progress)

    def _prepare_offline_dataset(self) -> None:
        if self.legacy_mode or not self.config.offline_rl.enabled:
            return
        if not self.offline_dataset_path.exists():
            export_offline_transition_dataset(
                self.dataset,
                self.offline_dataset_path,
                self.config,
                episodes=self.config.offline_rl.dataset_episodes,
                seed=self.config.train.seed,
            )
        self.offline_dataset = OfflineTransitionDataset.load(self.offline_dataset_path)
        self.offline_replay = replay_from_arrays(self.offline_dataset.arrays, self.device)
        if self.normalizer is not None:
            self.normalizer.update(
                {
                    "local_obs": self.offline_replay.local_obs,
                    "fleet_signature": self.offline_replay.fleet_signature,
                    "temporal_history": self.offline_replay.temporal_history,
                }
            )
    def _normalize_observation(self, observation: dict[str, Any], update_stats: bool) -> dict[str, torch.Tensor]:
        tensor_observation = ensure_tensor_observation(observation, self.device)
        if self.normalizer is None:
            return tensor_observation
        normalized = self.normalizer.normalize(
            {
                "local_obs": tensor_observation["local_obs"],
                "fleet_signature": tensor_observation["fleet_signature"],
                "temporal_history": tensor_observation["temporal_history"],
            },
            update_stats=update_stats,
        )
        normalized["agent_mask"] = tensor_observation["agent_mask"]
        normalized["action_mask"] = tensor_observation["action_mask"]
        return normalized

    def _normalize_replay_batch(self, batch: ReplayBatch) -> dict[str, torch.Tensor]:
        observation = {
            "local_obs": batch.local_obs,
            "fleet_signature": batch.fleet_signature,
            "temporal_history": batch.temporal_history,
        }
        if self.normalizer is not None:
            observation = self.normalizer.normalize(observation, update_stats=False)
        observation["agent_mask"] = batch.agent_mask
        observation["action_mask"] = batch.action_mask
        return observation

    def _critic_value(self, observation: dict[str, np.ndarray]) -> float:
        if self.legacy_mode:
            value_tensor = self.critic(
                torch.as_tensor(observation["local_obs"], dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.as_tensor(observation["fleet_signature"], dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.as_tensor(observation["agent_mask"], dtype=torch.float32, device=self.device).unsqueeze(0),
            )
            return float(value_tensor.item())
        normalized = self._normalize_observation(observation, update_stats=False)
        return float(self.critic(normalized).mean.item())

    def _critic_uncertainty_std(self, observation: dict[str, np.ndarray]) -> float:
        if self.legacy_mode or not isinstance(self.critic, EnsembleCritic):
            return 0.0
        normalized = self._normalize_observation(observation, update_stats=False)
        with torch.no_grad():
            critic_output = self.critic(normalized)
        return float(torch.sqrt(critic_output.variance.mean() + 1e-8).item())

    def _make_action_selector(self, execution_mode: str, current_episode: int) -> Any:
        def _select(observation: dict[str, Any]) -> Any:
            actions, _, planner_info = action_selection_from_policy(
                self.actor,
                observation,
                self.device,
                critic=self.critic if isinstance(self.critic, EnsembleCritic) else None,
                cost_critic=self.cost_critic,
                planner=self.planner,
                env=self.env,
                normalizer=self.normalizer,
                planner_enabled=self.config.online_finetune.planner_enabled,
                execution_mode=execution_mode,
                uncertainty_calibrator=self.uncertainty_calibrator,
                current_episode=current_episode,
            )
            active_agents = int(np.asarray(observation["agent_mask"]).sum())
            self.env.register_runtime_feedback(
                policy_count=int(np.asarray(planner_info["policy_selected_mask"]).sum()),
                planner_count=int(np.asarray(planner_info["planner_selected_mask"]).sum()),
                fallback_count=int(np.asarray(planner_info["fallback_mask"]).sum()),
                uncertainty_count=int(np.asarray(planner_info["uncertainty_mask"]).sum()),
                decision_steps=active_agents,
            )
            return actions

        return _select

    def _run_rollout(self, episode: int) -> tuple[RolloutBatch, dict[str, float]]:
        self.env.set_training_progress(episode, self.config.train.total_episodes)
        observation = self.env.reset("train")
        buffer = RolloutBuffer()
        done = False
        final_info: dict[str, float] = {}
        while not done:
            if self.normalizer is not None:
                self.normalizer.update(observation)
            if not self.legacy_mode:
                self.uncertainty_calibrator.update(self._critic_uncertainty_std(observation))
            actions, log_probs, planner_info = action_selection_from_policy(
                self.actor,
                observation,
                self.device,
                critic=self.critic if isinstance(self.critic, EnsembleCritic) else None,
                cost_critic=self.cost_critic,
                planner=self.planner,
                env=self.env,
                normalizer=self.normalizer,
                planner_enabled=self.config.online_finetune.planner_enabled,
                execution_mode=self.config.train.execution_mode,
                uncertainty_calibrator=self.uncertainty_calibrator,
                current_episode=episode,
            )
            self.env.register_runtime_feedback(
                policy_count=int(np.asarray(planner_info["policy_selected_mask"]).sum()),
                planner_count=int(np.asarray(planner_info["planner_selected_mask"]).sum()),
                fallback_count=int(np.asarray(planner_info["fallback_mask"]).sum()),
                uncertainty_count=int(np.asarray(planner_info["uncertainty_mask"]).sum()),
                decision_steps=int(np.asarray(observation["agent_mask"]).sum()),
            )
            value = self._critic_value(observation)
            next_observation, _, team_reward, done, info = self.env.step(actions)
            buffer.add(
                observation=observation,
                actions=actions,
                log_probs=log_probs,
                value=value,
                reward=team_reward,
                done=done,
                costs=info["costs"],
                aux_targets=info["aux_targets"],
                uncertainty=float(planner_info["uncertainty"]),
            )
            self.online_replay.add(
                {
                    "local_obs": observation["local_obs"],
                    "fleet_signature": observation["fleet_signature"],
                    "temporal_history": observation["temporal_history"],
                    "agent_mask": observation["agent_mask"],
                    "action_mask": observation["action_mask"],
                    "actions": actions,
                    "rewards": team_reward,
                    "returns": team_reward,
                    "costs": np.asarray([info["costs"][name] for name in COST_NAMES], dtype=np.float32),
                    "next_local_obs": next_observation["local_obs"],
                    "next_fleet_signature": next_observation["fleet_signature"],
                    "next_temporal_history": next_observation["temporal_history"],
                    "next_agent_mask": next_observation["agent_mask"],
                    "next_action_mask": next_observation["action_mask"],
                    "dones": float(done),
                    "aux_next_demand": info["aux_targets"]["next_demand"],
                    "aux_charger_occupancy": info["aux_targets"]["charger_occupancy"],
                    "aux_travel_time_residual": np.asarray([info["aux_targets"]["travel_time_residual"]], dtype=np.float32),
                }
            )
            observation = next_observation
            final_info = info

        batch = buffer.compute_batch(
            last_value=0.0,
            gamma=self.config.train.gamma,
            gae_lambda=self.config.train.gae_lambda,
            device=self.device,
        )
        return batch, final_info

    def _supervised_replay_update(self, batch: ReplayBatch, mode: str) -> dict[str, float]:
        normalized = self._normalize_replay_batch(batch)
        critic_output = self.critic(normalized)
        actor_output = self.actor(normalized)
        log_probs, _ = policy_statistics(
            actor_output.logits,
            batch.action_mask,
            batch.actions,
            batch.agent_mask,
        )
        centered_returns = batch.returns - batch.returns.mean()
        if mode == "awac":
            weights = torch.exp(centered_returns / max(self.config.offline_rl.awac_lambda, 1e-6)).clamp(max=20.0)
        else:
            weights = torch.exp(centered_returns / max(self.config.offline_rl.temperature, 1e-6)).clamp(max=20.0)
        actor_loss = -((weights.unsqueeze(-1) * log_probs).sum() / batch.agent_mask.sum().clamp_min(1.0))
        if mode == "cql":
            actor_loss = actor_loss + self.config.offline_rl.cql_alpha * torch.logsumexp(actor_output.logits, dim=-1).mean()
        value_loss = torch.mean((critic_output.mean - batch.returns) ** 2)
        cost_value_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.cost_critic is not None:
            cost_outputs = self.cost_critic(normalized)
            cost_value_loss = sum(
                torch.mean((cost_outputs[name].mean - batch.costs[:, index]) ** 2)
                for index, name in enumerate(COST_NAMES)
            )
        aux_loss = (
            torch.mean((actor_output.aux_predictions["next_demand"] - batch.aux_next_demand) ** 2)
            + torch.mean((actor_output.aux_predictions["charger_occupancy"] - batch.aux_charger_occupancy) ** 2)
            + torch.mean((actor_output.aux_predictions["travel_time_residual"] - batch.aux_travel_time_residual) ** 2)
        )
        total_loss = actor_loss + self.config.train.value_coef * value_loss + self.config.train.aux_coef * aux_loss
        if self.cost_critic is not None:
            total_loss = total_loss + self.config.train.cost_value_coef * cost_value_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        clip_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.cost_critic is not None:
            clip_parameters += list(self.cost_critic.parameters())
        nn.utils.clip_grad_norm_(clip_parameters, self.config.train.max_grad_norm)
        self.optimizer.step()
        return {
            "offline_actor_loss": float(actor_loss.item()),
            "offline_value_loss": float(value_loss.item()),
            "offline_aux_loss": float(aux_loss.item()),
        }

    def _offline_pretrain(self) -> dict[str, float]:
        if self.legacy_mode or not self.config.offline_rl.enabled:
            return {"offline_actor_loss": 0.0, "offline_value_loss": 0.0, "offline_aux_loss": 0.0}
        self._prepare_offline_dataset()
        assert self.offline_replay is not None
        aggregate = {"offline_actor_loss": 0.0, "offline_value_loss": 0.0, "offline_aux_loss": 0.0, "updates": 0.0}
        indices = np.arange(self.offline_replay.returns.shape[0])
        rng = np.random.default_rng(self.config.train.seed)
        for _ in range(self.config.offline_rl.pretrain_epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), self.config.offline_rl.batch_size):
                mb_indices = indices[start : start + self.config.offline_rl.batch_size]
                if len(mb_indices) == 0:
                    continue
                mb = self._slice_replay_batch(self.offline_replay, mb_indices)
                losses = self._supervised_replay_update(mb, mode=self.config.offline_rl.algorithm)
                for key in ("offline_actor_loss", "offline_value_loss", "offline_aux_loss"):
                    aggregate[key] += losses[key]
                aggregate["updates"] += 1.0
        if aggregate["updates"] > 0:
            for key in ("offline_actor_loss", "offline_value_loss", "offline_aux_loss"):
                aggregate[key] /= aggregate["updates"]
        return {key: value for key, value in aggregate.items() if key != "updates"}
    def _update_online(self, batch: RolloutBatch) -> dict[str, float]:
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        total_steps = batch.returns.shape[0]
        indices = np.arange(total_steps)
        rng = np.random.default_rng(self.config.train.seed)
        aggregate = {
            "actor_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "aux_loss": 0.0,
            "cost_value_loss": 0.0,
            "updates": 0.0,
        }

        for _ in range(self.config.train.ppo_epochs):
            rng.shuffle(indices)
            for start in range(0, total_steps, self.config.train.minibatch_size):
                mb_indices = indices[start : start + self.config.train.minibatch_size]
                if len(mb_indices) == 0:
                    continue
                if self.legacy_mode:
                    logits, aux_prediction = self.actor(
                        batch.local_obs[mb_indices], batch.fleet_signature[mb_indices]
                    )
                    values = self.critic(
                        batch.local_obs[mb_indices],
                        batch.fleet_signature[mb_indices],
                        batch.agent_mask[mb_indices],
                    )
                    aux_loss = torch.mean((aux_prediction - batch.aux_next_demand[mb_indices]) ** 2)
                    cost_value_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                else:
                    normalized = self.normalizer.normalize(
                        {
                            "local_obs": batch.local_obs[mb_indices],
                            "fleet_signature": batch.fleet_signature[mb_indices],
                            "temporal_history": batch.temporal_history[mb_indices],
                        },
                        update_stats=False,
                    )
                    normalized["agent_mask"] = batch.agent_mask[mb_indices]
                    normalized["action_mask"] = batch.action_mask[mb_indices]
                    actor_output = self.actor(normalized)
                    logits = actor_output.logits
                    values = self.critic(normalized).mean
                    aux_loss = (
                        torch.mean((actor_output.aux_predictions["next_demand"] - batch.aux_next_demand[mb_indices]) ** 2)
                        + torch.mean((actor_output.aux_predictions["charger_occupancy"] - batch.aux_charger_occupancy[mb_indices]) ** 2)
                        + torch.mean((actor_output.aux_predictions["travel_time_residual"] - batch.aux_travel_time_residual[mb_indices]) ** 2)
                    )
                    cost_outputs = self.cost_critic(normalized)
                    cost_value_loss = sum(
                        torch.mean((cost_outputs[name].mean - batch.cost_returns[mb_indices, index]) ** 2)
                        for index, name in enumerate(COST_NAMES)
                    )
                log_probs, entropy = policy_statistics(
                    logits,
                    batch.action_mask[mb_indices],
                    batch.actions[mb_indices],
                    batch.agent_mask[mb_indices],
                )
                old_log_probs = batch.old_log_probs[mb_indices]
                ratio = torch.exp(log_probs - old_log_probs)
                combined_advantages = advantages[mb_indices]
                if not self.legacy_mode:
                    for cost_index, name in enumerate(COST_NAMES):
                        combined_advantages = combined_advantages - self.constraint_multipliers[name] * batch.cost_advantages[mb_indices, cost_index]
                adv = combined_advantages.unsqueeze(-1)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.train.ppo_clip,
                    1.0 + self.config.train.ppo_clip,
                ) * adv
                agent_mask = batch.agent_mask[mb_indices]
                active_count = agent_mask.sum().clamp_min(1.0)
                actor_loss = -(torch.minimum(surr1, surr2) * agent_mask).sum() / active_count
                entropy_bonus = (entropy * agent_mask).sum() / active_count
                value_loss = torch.mean((values - batch.returns[mb_indices]) ** 2)
                total_loss = (
                    actor_loss
                    + self.config.train.value_coef * value_loss
                    + self.config.train.cost_value_coef * cost_value_loss
                    - self.config.train.entropy_coef * entropy_bonus
                    + self.config.train.aux_coef * aux_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                clip_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
                if self.cost_critic is not None:
                    clip_parameters += list(self.cost_critic.parameters())
                nn.utils.clip_grad_norm_(clip_parameters, self.config.train.max_grad_norm)
                self.optimizer.step()

                aggregate["actor_loss"] += float(actor_loss.item())
                aggregate["value_loss"] += float(value_loss.item())
                aggregate["entropy"] += float(entropy_bonus.item())
                aggregate["aux_loss"] += float(aux_loss.item())
                aggregate["cost_value_loss"] += float(cost_value_loss.item())
                aggregate["updates"] += 1.0

        if aggregate["updates"] > 0:
            for key in ("actor_loss", "value_loss", "entropy", "aux_loss", "cost_value_loss"):
                aggregate[key] /= aggregate["updates"]
        return {key: value for key, value in aggregate.items() if key != "updates"}

    def _mixed_replay_update(self) -> dict[str, float]:
        ratio = self._offline_ratio() if not self.legacy_mode else 0.0
        if self.legacy_mode or self.offline_replay is None or len(self.online_replay) == 0:
            return {"offline_ratio": ratio, "online_ratio": 1.0 - ratio, "mixed_actor_loss": 0.0}
        batch_size = self.config.offline_rl.batch_size
        offline_count = max(1, int(batch_size * ratio))
        offline_indices = np.random.choice(
            self.offline_replay.returns.shape[0],
            size=min(offline_count, self.offline_replay.returns.shape[0]),
            replace=False,
        )
        offline_batch = self._slice_replay_batch(self.offline_replay, offline_indices)
        online_batch = self.online_replay.sample(max(1, batch_size - len(offline_indices)), self.device)
        replay_batch = ReplayBatch(
            local_obs=torch.cat([offline_batch.local_obs, online_batch.local_obs], dim=0),
            fleet_signature=torch.cat([offline_batch.fleet_signature, online_batch.fleet_signature], dim=0),
            temporal_history=torch.cat([offline_batch.temporal_history, online_batch.temporal_history], dim=0),
            agent_mask=torch.cat([offline_batch.agent_mask, online_batch.agent_mask], dim=0),
            action_mask=torch.cat([offline_batch.action_mask, online_batch.action_mask], dim=0),
            actions=torch.cat([offline_batch.actions, online_batch.actions], dim=0),
            rewards=torch.cat([offline_batch.rewards, online_batch.rewards], dim=0),
            returns=torch.cat([offline_batch.returns, online_batch.returns], dim=0),
            costs=torch.cat([offline_batch.costs, online_batch.costs], dim=0),
            next_local_obs=torch.cat([offline_batch.next_local_obs, online_batch.next_local_obs], dim=0),
            next_fleet_signature=torch.cat([offline_batch.next_fleet_signature, online_batch.next_fleet_signature], dim=0),
            next_temporal_history=torch.cat([offline_batch.next_temporal_history, online_batch.next_temporal_history], dim=0),
            next_agent_mask=torch.cat([offline_batch.next_agent_mask, online_batch.next_agent_mask], dim=0),
            next_action_mask=torch.cat([offline_batch.next_action_mask, online_batch.next_action_mask], dim=0),
            dones=torch.cat([offline_batch.dones, online_batch.dones], dim=0),
            aux_next_demand=torch.cat([offline_batch.aux_next_demand, online_batch.aux_next_demand], dim=0),
            aux_charger_occupancy=torch.cat([offline_batch.aux_charger_occupancy, online_batch.aux_charger_occupancy], dim=0),
            aux_travel_time_residual=torch.cat([offline_batch.aux_travel_time_residual, online_batch.aux_travel_time_residual], dim=0),
        )
        losses = self._supervised_replay_update(replay_batch, mode=self.config.offline_rl.algorithm)
        return {
            "offline_ratio": ratio,
            "online_ratio": 1.0 - ratio,
            "mixed_actor_loss": losses["offline_actor_loss"],
        }

    def _update_multipliers(self, batch: RolloutBatch, episode: int) -> bool:
        if self.legacy_mode:
            return False
        if episode <= self.config.safety.constraint_warmup_episodes:
            return True
        observed = batch.costs.mean(dim=0).detach().cpu().numpy()
        limits = {
            "battery_violation_cost": self.config.safety.battery_limit,
            "charger_overflow_cost": self.config.safety.charger_limit,
            "service_violation_cost": self.config.safety.service_limit,
        }
        for index, name in enumerate(COST_NAMES):
            ema = (
                self.config.safety.multiplier_ema_decay * self.constraint_ema[name]
                + (1.0 - self.config.safety.multiplier_ema_decay) * float(observed[index])
            )
            self.constraint_ema[name] = float(ema)
            error = float(np.clip(ema - limits[name], -1.0, 1.0))
            updated = self.constraint_multipliers[name] + self.config.safety.multiplier_lr * error
            self.constraint_multipliers[name] = float(
                np.clip(updated, 0.0, self.config.safety.multiplier_max)
            )
        return False
    def train(self) -> pd.DataFrame:
        config_path = self.output_dir / "config_snapshot.json"
        config_path.write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        train_start = time.perf_counter()
        if self.start_episode == 1 and not self.history:
            offline_metrics = self._offline_pretrain()
            self.history.append({"episode": 0, **offline_metrics})

        cumulative_env_steps = 0
        for episode in range(self.start_episode, self.config.train.total_episodes + 1):
            batch, info = self._run_rollout(episode)
            cumulative_env_steps += int(batch.returns.shape[0])
            losses = self._update_online(batch)
            mixed_losses = self._mixed_replay_update()
            constraint_warmup_active = self._update_multipliers(batch, episode)
            self.scheduler_state["step"] += 1
            elapsed = self.resume_wall_clock_seconds + (time.perf_counter() - train_start)
            hours = max(elapsed / 3600.0, 1e-6)
            record: dict[str, float | int] = {
                "episode": episode,
                "train_mean_team_reward": float(info["mean_team_reward"]),
                "train_order_completion_rate": float(info["order_completion_rate"]),
                "train_average_profit_per_vehicle": float(info["average_profit_per_vehicle"]),
                "train_battery_violation_rate": float(info["battery_violation_rate"]),
                "train_charger_overflow_rate": float(info["charger_overflow_rate"]),
                "train_service_violation_rate": float(info["service_violation_rate"]),
                "train_service_utilization_rate": float(info["service_utilization_rate"]),
                "train_policy_selected_rate": float(info["policy_selected_rate"]),
                "train_planner_selected_rate": float(info["planner_selected_rate"]),
                "train_fallback_rate": float(info["fallback_rate"]),
                "train_uncertainty_trigger_rate": float(info["uncertainty_trigger_rate"]),
                "lambda_battery": self.constraint_multipliers["battery_violation_cost"],
                "lambda_charger": self.constraint_multipliers["charger_overflow_cost"],
                "lambda_service": self.constraint_multipliers["service_violation_cost"],
                "lambda_battery_ema": self.constraint_ema["battery_violation_cost"],
                "lambda_charger_ema": self.constraint_ema["charger_overflow_cost"],
                "lambda_service_ema": self.constraint_ema["service_violation_cost"],
                "constraint_warmup_active": float(constraint_warmup_active),
                "wall_clock_seconds": elapsed,
                "episodes_per_hour": float(max(episode - self.start_episode + 1, 1) / hours),
                "steps_per_second": float(cumulative_env_steps / max(elapsed, 1e-6)),
                "best_val_mean_team_reward": self.best_val_mean_team_reward,
                "best_runtime_mean_team_reward": self.best_runtime_mean_team_reward,
                "checkpoint_tag": "none",
                **losses,
                **mixed_losses,
            }

            best_policy_pending = False
            best_runtime_pending = False
            if episode % self.config.train.eval_interval == 0:
                policy_metrics_frame, _ = evaluate_policy(
                    self.env,
                    self._make_action_selector(self.config.train.validation_execution_mode, episode),
                    "val",
                    episodes=1,
                    scenario_name="policy_validation",
                )
                policy_mean_team_reward = None
                for key, value in policy_metrics_frame.iloc[0].to_dict().items():
                    if key in {"split", "episodes", "scenario", "execution_mode"}:
                        continue
                    record[f"eval_policy_{key}"] = float(value)
                    record[f"eval_{key}"] = float(value)
                    if key == "mean_team_reward":
                        policy_mean_team_reward = float(value)
                if policy_mean_team_reward is not None:
                    record["eval_mean_team_reward"] = policy_mean_team_reward
                if (
                    policy_mean_team_reward is not None
                    and policy_mean_team_reward > self.best_val_mean_team_reward
                ):
                    self.best_val_mean_team_reward = policy_mean_team_reward
                    record["best_val_mean_team_reward"] = self.best_val_mean_team_reward
                    best_policy_pending = True

                runtime_metrics_frame, _ = evaluate_policy(
                    self.env,
                    self._make_action_selector(self.config.planner.runtime_execution_mode, episode),
                    "val",
                    episodes=1,
                    scenario_name="runtime_validation",
                )
                runtime_mean_team_reward = None
                for key, value in runtime_metrics_frame.iloc[0].to_dict().items():
                    if key in {"split", "episodes", "scenario", "execution_mode"}:
                        continue
                    record[f"eval_runtime_{key}"] = float(value)
                    if key == "mean_team_reward":
                        runtime_mean_team_reward = float(value)
                if (
                    runtime_mean_team_reward is not None
                    and runtime_mean_team_reward > self.best_runtime_mean_team_reward
                ):
                    self.best_runtime_mean_team_reward = runtime_mean_team_reward
                    record["best_runtime_mean_team_reward"] = self.best_runtime_mean_team_reward
                    best_runtime_pending = True

            self.history.append(record)

            checkpoint_tag = "none"
            if episode % self.config.train.save_interval == 0 or episode == self.config.train.total_episodes:
                self._save_checkpoint(episode)
                checkpoint_tag = "periodic+latest"
                if best_policy_pending:
                    self._save_named_checkpoint(episode, "best_policy_val.pt")
                    self._save_named_checkpoint(episode, "best_val.pt")
                if best_runtime_pending:
                    self._save_named_checkpoint(episode, "best_runtime_val.pt")
                if best_policy_pending and best_runtime_pending:
                    checkpoint_tag = "best_policy+best_runtime+periodic+latest"
                elif best_policy_pending:
                    checkpoint_tag = "best_policy+periodic+latest"
                elif best_runtime_pending:
                    checkpoint_tag = "best_runtime+periodic+latest"
            elif best_policy_pending or best_runtime_pending:
                self._save_checkpoint(episode)
                if best_policy_pending:
                    self._save_named_checkpoint(episode, "best_policy_val.pt")
                    self._save_named_checkpoint(episode, "best_val.pt")
                if best_runtime_pending:
                    self._save_named_checkpoint(episode, "best_runtime_val.pt")
                if best_policy_pending and best_runtime_pending:
                    checkpoint_tag = "best_policy+best_runtime+latest"
                elif best_policy_pending:
                    checkpoint_tag = "best_policy+latest"
                else:
                    checkpoint_tag = "best_runtime+latest"
            record["best_val_mean_team_reward"] = self.best_val_mean_team_reward
            record["best_runtime_mean_team_reward"] = self.best_runtime_mean_team_reward
            record["checkpoint_tag"] = checkpoint_tag
            self.resume_wall_clock_seconds = elapsed

        history_frame = pd.DataFrame(self.history)
        history_frame.to_csv(self.output_dir / "metrics.csv", index=False)
        write_reward_curve(history_frame, self.output_dir)
        return history_frame
