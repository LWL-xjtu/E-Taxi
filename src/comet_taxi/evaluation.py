from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "comet_taxi_mpl")
)

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .baselines import GreedyDispatchPolicy
from .config import ExperimentConfig
from .env import CometTaxiEnv
from .models import COMETActor
from .runtime import action_selection_from_policy, resolve_device
from .utils import ensure_dir


def evaluate_policy(
    env: CometTaxiEnv,
    select_actions: Callable[[dict[str, object]], object],
    split: str,
    episodes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, float | int | str]] = []
    for episode in range(episodes):
        observation = env.reset(split=split)
        done = False
        while not done:
            actions = select_actions(observation)
            observation, _, _, done, info = env.step(actions)
        summary = {"episode": episode + 1, **info}
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries)
    metrics_frame = pd.DataFrame(
        [
            {
                "split": split,
                "episodes": episodes,
                **summary_frame.drop(columns=["episode"]).mean(numeric_only=True).to_dict(),
            }
        ]
    )
    return metrics_frame, summary_frame


def evaluate_checkpoint(
    config: ExperimentConfig,
    env: CometTaxiEnv,
    actor: COMETActor,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    episodes: int = 1,
) -> pd.DataFrame:
    device = resolve_device(config.train.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.to(device)
    actor.eval()

    def select_actions(observation: dict[str, object]) -> object:
        return action_selection_from_policy(actor, observation, device)[0]

    metrics_frame, summary_frame = evaluate_policy(env, select_actions, split, episodes)
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
) -> pd.DataFrame:
    policy = GreedyDispatchPolicy(config)
    metrics_frame, summary_frame = evaluate_policy(env, policy.act, split, episodes)
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
