from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "comet_taxi_mpl")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .baselines import GreedyDispatchPolicy
from .config import ExperimentConfig
from .env import CometTaxiEnv
from .models import COMETActor, COMETActorV2, CostCritic, EnsembleCritic, ObservationNormalizer
from .planner import CandidatePlanner
from .runtime import action_selection_from_policy, resolve_device
from .utils import ensure_dir


def evaluate_policy(
    env: CometTaxiEnv,
    select_actions: Callable[[dict[str, object]], object],
    split: str,
    episodes: int,
    scenario_name: str = "standard",
    scenario: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, float | int | str]] = []
    for episode in range(episodes):
        observation = env.reset(split=split, scenario=scenario)
        done = False
        while not done:
            actions = select_actions(observation)
            observation, _, _, done, info = env.step(actions)
        summary = {"episode": episode + 1, "scenario": scenario_name, **info}
        if "costs" in summary:
            for key, value in dict(summary["costs"]).items():
                summary[key] = value
            del summary["costs"]
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries)
    metrics_frame = pd.DataFrame(
        [
            {
                "split": split,
                "episodes": episodes,
                "scenario": scenario_name,
                **summary_frame.drop(columns=["episode"]).mean(numeric_only=True).to_dict(),
            }
        ]
    )
    return metrics_frame, summary_frame


def build_stress_scenarios(config: ExperimentConfig) -> list[tuple[str, dict[str, float]]]:
    scenarios: list[tuple[str, dict[str, float]]] = [("standard_test", {})]
    for fleet_size in config.evaluation.unseen_fleet_sizes:
        scenarios.append((f"unseen_fleet_{fleet_size}", {"active_agents": float(fleet_size)}))
    for outage in config.evaluation.charger_outage_levels:
        scenarios.append((f"charger_outage_{outage:.2f}", {"charger_capacity_scale": max(0.1, 1.0 - float(outage))}))
    for shock in config.evaluation.demand_shock_levels:
        scenarios.append((f"demand_shock_{shock:.2f}", {"demand_scale": float(shock)}))
    for shock in config.evaluation.travel_time_shock_levels:
        scenarios.append((f"travel_time_{shock:.2f}", {"travel_noise": float(shock)}))
    if config.evaluation.event_day_presets:
        scenarios.append(
            (
                "mixed_ood_stress",
                {
                    "demand_scale": float(max(config.evaluation.demand_shock_levels)),
                    "travel_noise": float(max(config.evaluation.travel_time_shock_levels)),
                    "charger_capacity_scale": max(0.1, 1.0 - float(max(config.evaluation.charger_outage_levels))),
                    "event_scale": 1.4,
                },
            )
        )
    return scenarios


def evaluate_checkpoint(
    config: ExperimentConfig,
    env: CometTaxiEnv,
    actor: COMETActor | COMETActorV2,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    episodes: int = 1,
    critic: EnsembleCritic | None = None,
    cost_critic: CostCritic | None = None,
    normalizer: ObservationNormalizer | None = None,
    stress: bool = False,
) -> pd.DataFrame:
    device = resolve_device(config.train.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.to(device)
    actor.eval()
    planner = CandidatePlanner(config)
    if critic is not None and "critic_state_dict" in checkpoint:
        critic.load_state_dict(checkpoint["critic_state_dict"])
        critic.to(device)
        critic.eval()
    if cost_critic is not None and "cost_critic_state_dict" in checkpoint:
        cost_critic.load_state_dict(checkpoint["cost_critic_state_dict"])
        cost_critic.to(device)
        cost_critic.eval()
    if normalizer is not None and checkpoint.get("normalizer_state_dict"):
        normalizer.load_state_dict(checkpoint["normalizer_state_dict"])

    def select_actions(observation: dict[str, object]) -> object:
        actions, _, planner_info = action_selection_from_policy(
            actor,
            observation,
            device,
            critic=critic,
            cost_critic=cost_critic,
            planner=planner,
            env=env,
            normalizer=normalizer,
            planner_enabled=config.online_finetune.planner_enabled,
        )
        env.register_runtime_feedback(
            fallback_count=int(np.asarray(planner_info["fallback_mask"]).sum()),
            uncertainty_count=int(np.asarray(planner_info["uncertainty_mask"]).sum()),
            planner_steps=int(np.asarray(observation["agent_mask"]).sum()),
        )
        return actions

    scenario_specs = build_stress_scenarios(config) if stress else [("standard_test", {})]
    metrics_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for scenario_name, scenario in scenario_specs:
        metrics_frame, summary_frame = evaluate_policy(
            env,
            select_actions,
            split,
            episodes,
            scenario_name=scenario_name,
            scenario=scenario,
        )
        metrics_frames.append(metrics_frame)
        summary_frames.append(summary_frame)
    metrics_frame = pd.concat(metrics_frames, ignore_index=True)
    summary_frame = pd.concat(summary_frames, ignore_index=True)
    output_root = ensure_dir(output_dir)
    metrics_frame.to_csv(output_root / "metrics.csv", index=False)
    summary_frame.to_csv(output_root / "episode_summaries.csv", index=False)
    return metrics_frame


def evaluate_greedy(
    config: ExperimentConfig,
    env: CometTaxiEnv,
    output_dir: str | Path,
    split: str = "test",
    episodes: int = 1,
    stress: bool = False,
) -> pd.DataFrame:
    policy = GreedyDispatchPolicy(config)
    scenario_specs = build_stress_scenarios(config) if stress else [("standard_test", {})]
    metrics_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for scenario_name, scenario in scenario_specs:
        metrics_frame, summary_frame = evaluate_policy(
            env,
            policy.act,
            split,
            episodes,
            scenario_name=scenario_name,
            scenario=scenario,
        )
        metrics_frames.append(metrics_frame)
        summary_frames.append(summary_frame)
    metrics_frame = pd.concat(metrics_frames, ignore_index=True)
    summary_frame = pd.concat(summary_frames, ignore_index=True)
    output_root = ensure_dir(output_dir)
    metrics_frame.to_csv(output_root / "metrics.csv", index=False)
    summary_frame.to_csv(output_root / "episode_summaries.csv", index=False)
    return metrics_frame


def write_reward_curve(history: pd.DataFrame, output_dir: str | Path) -> None:
    output_root = ensure_dir(output_dir)
    history.to_csv(output_root / "reward_curve.csv", index=False)
    plt.figure(figsize=(8, 4))
    plt.plot(history["episode"], history["train_mean_team_reward"], label="train")
    if "eval_mean_team_reward" in history:
        plt.plot(history["episode"], history["eval_mean_team_reward"], label="eval")
    plt.xlabel("Episode")
    plt.ylabel("Mean Team Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "reward_curve.png")
    plt.close()
