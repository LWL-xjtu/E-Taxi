from __future__ import annotations

from pathlib import Path

import pandas as pd

from comet_taxi.visualize import (
    build_eval_comparison,
    plot_constraint_dashboard,
    plot_fallback_dashboard,
    plot_training_dashboard,
)


def test_visualization_writes_expected_assets(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    output_dir = tmp_path / "viz"
    train_dir.mkdir()
    eval_dir.mkdir()

    pd.DataFrame(
        [
            {
                "episode": 0,
                "offline_actor_loss": 0.2,
                "offline_value_loss": 0.3,
                "offline_aux_loss": 0.1,
            },
            {
                "episode": 1,
                "train_mean_team_reward": 0.2,
                "train_order_completion_rate": 0.4,
                "train_average_profit_per_vehicle": 3.0,
                "actor_loss": 0.1,
                "value_loss": 0.2,
                "entropy": 0.3,
                "aux_loss": 0.05,
                "cost_value_loss": 0.03,
                "eval_mean_team_reward": 0.15,
                "eval_order_completion_rate": 0.35,
                "eval_average_profit_per_vehicle": 2.8,
                "eval_empty_travel_ratio": 0.22,
                "eval_battery_safety_rate": 0.9,
                "eval_charging_efficiency": 0.8,
            },
        ]
    ).to_csv(train_dir / "metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "split": "test",
                "episodes": 1,
                "scenario": "standard_test",
                "mean_team_reward": 0.25,
                "order_completion_rate": 0.45,
                "average_profit_per_vehicle": 3.2,
                "empty_travel_ratio": 0.21,
                "battery_safety_rate": 0.93,
                "charging_efficiency": 0.84,
                "battery_violation_rate": 0.04,
                "charger_overflow_rate": 0.02,
                "service_violation_rate": 0.1,
                "fallback_rate": 0.05,
                "uncertainty_trigger_rate": 0.06,
            },
            {
                "split": "test",
                "episodes": 1,
                "scenario": "mixed_ood_stress",
                "mean_team_reward": 0.15,
                "order_completion_rate": 0.35,
                "average_profit_per_vehicle": 2.7,
                "empty_travel_ratio": 0.28,
                "battery_safety_rate": 0.88,
                "charging_efficiency": 0.72,
                "battery_violation_rate": 0.08,
                "charger_overflow_rate": 0.07,
                "service_violation_rate": 0.18,
                "fallback_rate": 0.12,
                "uncertainty_trigger_rate": 0.15,
            },
        ]
    ).to_csv(eval_dir / "metrics.csv", index=False)

    history = pd.read_csv(train_dir / "metrics.csv")
    plot_training_dashboard(history, output_dir)
    comparison = build_eval_comparison([eval_dir], ["COMET-v2"], output_dir)
    plot_constraint_dashboard(comparison, output_dir)
    plot_fallback_dashboard(comparison, output_dir)

    assert not comparison.empty
    assert (output_dir / "training_dashboard.png").exists()
    assert (output_dir / "loss_dashboard.png").exists()
    assert (output_dir / "validation_dashboard.png").exists()
    assert (output_dir / "comparison_dashboard.png").exists()
    assert (output_dir / "comparison_metrics.csv").exists()
    assert (output_dir / "robustness_dashboard.png").exists()
    assert (output_dir / "constraint_dashboard.png").exists()
    assert (output_dir / "fallback_dashboard.png").exists()
