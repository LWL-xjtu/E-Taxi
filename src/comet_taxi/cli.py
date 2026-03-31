from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_experiment_config
from .data import PreparedDataset, prepare_nyc_dataset
from .env import CometTaxiEnv
from .evaluation import evaluate_checkpoint, evaluate_greedy
from .models import (
    COMETActor,
    COMETActorV2,
    CostCritic,
    EnsembleCritic,
    ObservationNormalizer,
    infer_model_dimensions,
)
from .trainer import CometTrainer


def prepare_nyc_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the NYC TLC small-sample dataset.")
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    prepare_nyc_dataset(
        args.input,
        args.output,
        config.data,
        charge_station_count=config.env.charge_station_count,
    )


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train COMET / COMET-v2.")
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    trainer = CometTrainer(config, args.data_dir, args.output_dir)
    trainer.train()


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained COMET checkpoint.")
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    dataset = PreparedDataset.load(args.data_dir)
    env = CometTaxiEnv(dataset, config, seed=config.train.seed)
    dims = infer_model_dimensions(
        dataset.metadata["data"]["cell_count"],
        charger_count=len(dataset.metadata["charge_stations"]),
        history_len=config.temporal.history_len,
    )
    if config.model.variant == "legacy":
        actor = COMETActor(config.model, dims)
        critic = None
        cost_critic = None
        normalizer = None
    else:
        actor = COMETActorV2(config.model, config.set_encoder, config.temporal, dims)
        critic = EnsembleCritic(config.model, config.set_encoder, config.temporal, dims)
        cost_critic = CostCritic(config.model, config.set_encoder, config.temporal, dims)
        normalizer = ObservationNormalizer(dims)
    evaluate_checkpoint(
        config=config,
        env=env,
        actor=actor,
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        split=args.split,
        episodes=args.episodes,
        critic=critic,
        cost_critic=cost_critic,
        normalizer=normalizer,
        stress=args.stress,
    )


def run_greedy_baseline_main() -> None:
    parser = argparse.ArgumentParser(description="Run the greedy dispatch baseline.")
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    dataset = PreparedDataset.load(args.data_dir)
    env = CometTaxiEnv(dataset, config, seed=config.train.seed)
    evaluate_greedy(
        config=config,
        env=env,
        output_dir=Path(args.output_dir),
        split=args.split,
        episodes=args.episodes,
        stress=args.stress,
    )
