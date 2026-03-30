from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .buffer import RolloutBatch, RolloutBuffer
from .config import ExperimentConfig
from .data import PreparedDataset
from .env import CometTaxiEnv
from .evaluation import evaluate_policy, write_reward_curve
from .models import COMETActor, COMETCritic, infer_model_dimensions
from .runtime import action_selection_from_policy, policy_statistics, resolve_device
from .utils import ensure_dir, seed_everything


class CometTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        dataset_root: str | Path,
        output_dir: str | Path,
    ) -> None:
        self.config = config
        self.dataset = PreparedDataset.load(dataset_root)
        self.env = CometTaxiEnv(self.dataset, config, seed=config.train.seed)
        self.device = resolve_device(config.train.device)
        dims = infer_model_dimensions(self.dataset.metadata["data"]["cell_count"])
        self.actor = COMETActor(config.model, dims).to(self.device)
        self.critic = COMETCritic(config.model, dims).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.output_dir = ensure_dir(output_dir)
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.history: list[dict[str, float | int]] = []
        seed_everything(config.train.seed)

    def _save_checkpoint(self, episode: int) -> Path:
        checkpoint_path = self.checkpoint_dir / f"episode_{episode:04d}.pt"
        observation = self.env.reset("train")
        checkpoint = {
            "episode": episode,
            "config": self.config.to_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "normalization_stats": {
                "local_obs_mean": observation["local_obs"].mean(axis=0).tolist(),
                "local_obs_std": observation["local_obs"].std(axis=0).clip(min=1e-6).tolist(),
                "fleet_signature_mean": observation["fleet_signature"].tolist(),
                "fleet_signature_std": np.ones_like(observation["fleet_signature"]).tolist(),
            },
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def _run_rollout(self) -> tuple[RolloutBatch, dict[str, float]]:
        observation = self.env.reset("train")
        buffer = RolloutBuffer()
        done = False
        final_info: dict[str, float] = {}
        while not done:
            actions, log_probs = action_selection_from_policy(self.actor, observation, self.device)
            value_tensor = self.critic(
                torch.as_tensor(observation["local_obs"], dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.as_tensor(observation["fleet_signature"], dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.as_tensor(observation["agent_mask"], dtype=torch.float32, device=self.device).unsqueeze(0),
            )
            next_observation, _, team_reward, done, info = self.env.step(actions)
            buffer.add(
                observation=observation,
                actions=actions,
                log_probs=log_probs,
                value=float(value_tensor.item()),
                reward=team_reward,
                done=done,
                aux_target=info["next_demand_target"],
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

    def _update(self, batch: RolloutBatch) -> dict[str, float]:
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
            "updates": 0.0,
        }

        for _ in range(self.config.train.ppo_epochs):
            rng.shuffle(indices)
            for start in range(0, total_steps, self.config.train.minibatch_size):
                mb_indices = indices[start : start + self.config.train.minibatch_size]
                if len(mb_indices) == 0:
                    continue
                logits, aux_prediction = self.actor(
                    batch.local_obs[mb_indices], batch.fleet_signature[mb_indices]
                )
                values = self.critic(
                    batch.local_obs[mb_indices],
                    batch.fleet_signature[mb_indices],
                    batch.agent_mask[mb_indices],
                )
                log_probs, entropy = policy_statistics(
                    logits,
                    batch.action_mask[mb_indices],
                    batch.actions[mb_indices],
                    batch.agent_mask[mb_indices],
                )
                old_log_probs = batch.old_log_probs[mb_indices]
                ratio = torch.exp(log_probs - old_log_probs)
                adv = advantages[mb_indices].unsqueeze(-1)
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
                aux_loss = torch.mean((aux_prediction - batch.aux_targets[mb_indices]) ** 2)
                total_loss = (
                    actor_loss
                    + self.config.train.value_coef * value_loss
                    - self.config.train.entropy_coef * entropy_bonus
                    + self.config.train.aux_coef * aux_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.train.max_grad_norm,
                )
                self.optimizer.step()

                aggregate["actor_loss"] += float(actor_loss.item())
                aggregate["value_loss"] += float(value_loss.item())
                aggregate["entropy"] += float(entropy_bonus.item())
                aggregate["aux_loss"] += float(aux_loss.item())
                aggregate["updates"] += 1.0

        if aggregate["updates"] > 0:
            for key in ("actor_loss", "value_loss", "entropy", "aux_loss"):
                aggregate[key] /= aggregate["updates"]
        return aggregate

    def train(self) -> pd.DataFrame:
        config_path = self.output_dir / "config_snapshot.json"
        config_path.write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")

        def _eval_actions(observation: dict[str, Any]) -> Any:
            return action_selection_from_policy(self.actor, observation, self.device)[0]

        for episode in range(1, self.config.train.total_episodes + 1):
            batch, info = self._run_rollout()
            losses = self._update(batch)
            record: dict[str, float | int] = {
                "episode": episode,
                "train_mean_team_reward": float(info["mean_team_reward"]),
                "train_order_completion_rate": float(info["order_completion_rate"]),
                "train_average_profit_per_vehicle": float(info["average_profit_per_vehicle"]),
                **losses,
            }

            if episode % self.config.train.eval_interval == 0:
                metrics_frame, _ = evaluate_policy(self.env, _eval_actions, "val", episodes=1)
                for key, value in metrics_frame.iloc[0].to_dict().items():
                    if key in {"split", "episodes"}:
                        continue
                    record[f"eval_{key}"] = float(value)

            self.history.append(record)

            if episode % self.config.train.save_interval == 0 or episode == self.config.train.total_episodes:
                self._save_checkpoint(episode)

        history_frame = pd.DataFrame(self.history)
        history_frame.to_csv(self.output_dir / "metrics.csv", index=False)
        write_reward_curve(history_frame, self.output_dir)
        return history_frame
