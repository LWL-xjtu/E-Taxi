from __future__ import annotations

from pathlib import Path

import pandas as pd

from comet_taxi.visualize import build_eval_comparison, plot_training_dashboard


def test_visualization_writes_expected_assets(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    output_dir = tmp_path / "viz"
    train_dir.mkdir()
    eval_dir.mkdir()

    pd.DataFrame(
        [
            {
                "episode": 1,
                "train_mean_team_reward": 0.2,
                "train_order_completion_rate": 0.4,
                "train_average_profit_per_vehicle": 3.0,
                "actor_loss": 0.1,
                "value_loss": 0.2,
                "entropy": 0.3,
                "aux_loss": 0.05,
                "eval_mean_team_reward": 0.15,
                "eval_order_completion_rate": 0.35,
                "eval_average_profit_per_vehicle": 2.8,
                "eval_empty_travel_ratio": 0.22,
                "eval_battery_safety_rate": 0.9,
                "eval_charging_efficiency": 0.8,
            },
            {
                "episode": 2,
                "train_mean_team_reward": 0.3,
                "train_order_completion_rate": 0.5,
                "train_average_profit_per_vehicle": 3.5,
                "actor_loss": 0.08,
                "value_loss": 0.18,
                "entropy": 0.28,
                "aux_loss": 0.04,
                "eval_mean_team_reward": 0.2,
                "eval_order_completion_rate": 0.4,
                "eval_average_profit_per_vehicle": 3.1,
                "eval_empty_travel_ratio": 0.2,
                "eval_battery_safety_rate": 0.92,
                "eval_charging_efficiency": 0.82,
            },
        ]
    ).to_csv(train_dir / "metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "split": "test",
                "episodes": 1,
                "mean_team_reward": 0.25,
                "order_completion_rate": 0.45,
                "average_profit_per_vehicle": 3.2,
                "empty_travel_ratio": 0.21,
                "battery_safety_rate": 0.93,
                "charging_efficiency": 0.84,
            }
        ]
    ).to_csv(eval_dir / "metrics.csv", index=False)

    history = pd.read_csv(train_dir / "metrics.csv")
    plot_training_dashboard(history, output_dir)
    comparison = build_eval_comparison([eval_dir], ["COMET"], output_dir)

    assert not comparison.empty
    assert (output_dir / "training_dashboard.png").exists()
    assert (output_dir / "loss_dashboard.png").exists()
    assert (output_dir / "validation_dashboard.png").exists()
    assert (output_dir / "comparison_dashboard.png").exists()
    assert (output_dir / "comparison_metrics.csv").exists()
