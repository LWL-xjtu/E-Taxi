from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "comet_taxi_mpl")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir

TRAIN_PLOTS = [
    ("train_mean_team_reward", "Train Mean Team Reward"),
    ("train_order_completion_rate", "Train Order Completion Rate"),
    ("train_average_profit_per_vehicle", "Train Profit Per Vehicle"),
]

LOSS_PLOTS = [
    ("actor_loss", "Actor Loss"),
    ("value_loss", "Value Loss"),
    ("entropy", "Entropy Bonus"),
    ("aux_loss", "Auxiliary Loss"),
]

EVAL_METRICS = [
    ("mean_team_reward", "Mean Team Reward"),
    ("order_completion_rate", "Order Completion Rate"),
    ("average_profit_per_vehicle", "Profit Per Vehicle"),
    ("empty_travel_ratio", "Empty Travel Ratio"),
    ("battery_safety_rate", "Battery Safety Rate"),
    ("charging_efficiency", "Charging Efficiency"),
]

CONSTRAINT_METRICS = [
    ("battery_violation_rate", "Battery Violation Rate"),
    ("charger_overflow_rate", "Charger Overflow Rate"),
    ("service_violation_rate", "Service Violation Rate"),
]

FALLBACK_METRICS = [
    ("fallback_rate", "Fallback Rate"),
    ("uncertainty_trigger_rate", "Uncertainty Trigger Rate"),
]


def _load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_training_dashboard(history: pd.DataFrame, output_dir: str | Path) -> None:
    output_root = ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for axis, (column, title) in zip(axes, TRAIN_PLOTS):
        if column in history.columns:
            axis.plot(history["episode"], history[column], label="train", linewidth=2)
        eval_column = column.replace("train_", "eval_")
        if eval_column in history.columns:
            eval_history = history.dropna(subset=[eval_column])
            axis.plot(
                eval_history["episode"],
                eval_history[eval_column],
                label="eval",
                linestyle="--",
                linewidth=2,
            )
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_root / "training_dashboard.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    flat_axes = axes.flatten()
    for axis, (column, title) in zip(flat_axes, LOSS_PLOTS):
        if column in history.columns:
            axis.plot(history["episode"], history[column], linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_root / "loss_dashboard.png")
    plt.close(fig)

    eval_columns = [f"eval_{name}" for name, _ in EVAL_METRICS if f"eval_{name}" in history.columns]
    if eval_columns:
        eval_history = history.dropna(subset=eval_columns, how="all")
        if not eval_history.empty:
            fig, axes = plt.subplots(2, 3, figsize=(16, 8))
            for axis, (metric, title) in zip(axes.flatten(), EVAL_METRICS):
                column = f"eval_{metric}"
                if column in eval_history.columns:
                    axis.plot(eval_history["episode"], eval_history[column], linewidth=2)
                axis.set_title(f"Validation {title}")
                axis.set_xlabel("Episode")
                axis.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_root / "validation_dashboard.png")
            plt.close(fig)


def plot_episode_summaries(summary_frame: pd.DataFrame, output_dir: str | Path, prefix: str) -> None:
    output_root = ensure_dir(output_dir)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for axis, (metric, title) in zip(axes.flatten(), EVAL_METRICS):
        if metric in summary_frame.columns:
            axis.plot(summary_frame["episode"], summary_frame[metric], marker="o", linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_root / f"{prefix}_episode_dashboard.png")
    plt.close(fig)


def build_eval_comparison(
    eval_dirs: list[str | Path],
    labels: list[str] | None,
    output_dir: str | Path,
) -> pd.DataFrame:
    output_root = ensure_dir(output_dir)
    resolved_labels = labels or [Path(path).name for path in eval_dirs]
    records: list[dict[str, float | str]] = []
    for label, eval_dir in zip(resolved_labels, eval_dirs):
        metrics_frame = _load_csv(Path(eval_dir) / "metrics.csv")
        for _, row in metrics_frame.iterrows():
            record = {"label": label, "scenario": row.get("scenario", "standard_test")}
            for metric, _ in EVAL_METRICS + CONSTRAINT_METRICS + FALLBACK_METRICS:
                record[metric] = float(row.get(metric, 0.0))
            records.append(record)

    comparison = pd.DataFrame(records)
    comparison.to_csv(output_root / "comparison_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for axis, (metric, title) in zip(axes.flatten(), EVAL_METRICS):
        standard = comparison[comparison["scenario"] == "standard_test"]
        axis.bar(standard["label"], standard[metric])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_root / "comparison_dashboard.png")
    plt.close(fig)

    if "scenario" in comparison.columns:
        robustness = comparison.pivot_table(
            index="scenario",
            columns="label",
            values="mean_team_reward",
            aggfunc="mean",
        )
        robustness.plot(kind="bar", figsize=(12, 5))
        plt.ylabel("Mean Team Reward")
        plt.tight_layout()
        plt.savefig(output_root / "robustness_dashboard.png")
        plt.close()
    return comparison


def plot_constraint_dashboard(comparison: pd.DataFrame, output_dir: str | Path) -> None:
    output_root = ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (metric, title) in zip(axes, CONSTRAINT_METRICS):
        if metric in comparison.columns:
            axis.bar(comparison["label"], comparison[metric])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_root / "constraint_dashboard.png")
    plt.close(fig)


def plot_fallback_dashboard(comparison: pd.DataFrame, output_dir: str | Path) -> None:
    output_root = ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (metric, title) in zip(axes, FALLBACK_METRICS):
        if metric in comparison.columns:
            axis.bar(comparison["label"], comparison[metric])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_root / "fallback_dashboard.png")
    plt.close(fig)


def visualize_results_main() -> None:
    parser = argparse.ArgumentParser(description="Visualize COMET training and evaluation outputs.")
    parser.add_argument("--train-dir", help="Training output directory containing metrics.csv", default=None)
    parser.add_argument(
        "--eval-dirs",
        nargs="*",
        default=None,
        help="One or more evaluation output directories containing metrics.csv",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for --eval-dirs in the same order",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write visualization assets")
    args = parser.parse_args()

    output_root = ensure_dir(args.output_dir)

    if args.train_dir:
        train_dir = Path(args.train_dir)
        history = _load_csv(train_dir / "metrics.csv")
        plot_training_dashboard(history, output_root)

    if args.eval_dirs:
        if args.labels and len(args.labels) != len(args.eval_dirs):
            raise ValueError("If --labels is provided, it must match the number of --eval-dirs.")
        comparison = build_eval_comparison(args.eval_dirs, args.labels, output_root)
        plot_constraint_dashboard(comparison, output_root)
        plot_fallback_dashboard(comparison, output_root)
        for index, eval_dir in enumerate(args.eval_dirs):
            eval_path = Path(eval_dir)
            summary_path = eval_path / "episode_summaries.csv"
            if summary_path.exists():
                label = args.labels[index] if args.labels else eval_path.name
                summaries = _load_csv(summary_path)
                plot_episode_summaries(summaries, output_root, label)


if __name__ == "__main__":
    visualize_results_main()
