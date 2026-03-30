# COMET Taxi

COMET Taxi is a small-sample research prototype for dynamic-agent E-taxi dispatch and charging. It implements the COMET idea from the planning document:

- fixed-size multi-agent observations with `Ghost + Mask`
- parameter-shared actor and centralized masked critic
- MAPPO training with mean team reward
- NYC TLC small-sample preprocessing and a grid-world simulator
- pure-RL improvements only: auxiliary demand prediction, domain randomization, and slot shuffling

## Environment

This repository now includes a local virtual environment at [`.venv`](D:/Codex%20Project/.venv).

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## GPU Note

The default train device is `auto`, so the code will use `CUDA` automatically when available. On your A100 server, you should be able to keep the config unchanged and train on GPU directly.

The Windows environment created here installed the default `torch` wheel. If your A100 server needs a CUDA-specific PyTorch wheel, reinstall `torch` inside the server venv before training, for example:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

## Install

Dependencies are already installed in the local `.venv`. For a fresh machine:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Data Preparation

The preprocessing entrypoint expects a NYC TLC Yellow Taxi parquet file. The README example uses `yellow_tripdata_2023-01.parquet`.

```powershell
prepare_nyc --config configs/base.toml --input path\to\yellow_tripdata_2023-01.parquet --output data\processed\nyc_small
```

What preprocessing does:

- takes the first 7 days
- uses the 08:00-20:00 window with 10-minute bins
- builds a fixed `8x8` grid
- writes `train.parquet`, `val.parquet`, `test.parquet`, and `metadata.json`

The code supports two input variants:

- coordinate columns (`pickup_longitude`, `pickup_latitude`, ...)
- modern TLC `PULocationID` / `DOLocationID`, projected deterministically onto the fixed grid

## Train

```powershell
train --config configs/base.toml --data-dir data\processed\nyc_small --output-dir outputs\comet_run
```

Outputs:

- `metrics.csv`
- `reward_curve.csv`
- `reward_curve.png`
- `checkpoints\episode_XXXX.pt`
- `config_snapshot.json`

## Evaluate

```powershell
evaluate --config configs/base.toml --data-dir data\processed\nyc_small --checkpoint outputs\comet_run\checkpoints\episode_0060.pt --output-dir outputs\eval_test --split test --episodes 1
```

## Greedy Baseline

```powershell
run_greedy_baseline --config configs/base.toml --data-dir data\processed\nyc_small --output-dir outputs\greedy_test --split test --episodes 1
```

## Visualization

Training outputs currently contain:

- `metrics.csv`: per-episode training metrics and PPO losses
- `reward_curve.csv`: training reward history
- `reward_curve.png`: a basic reward plot
- `checkpoints\episode_XXXX.pt`: saved actor/critic checkpoints
- `config_snapshot.json`: config used for the run

Evaluation outputs currently contain:

- `metrics.csv`: aggregated evaluation metrics
- `episode_summaries.csv`: per-episode metrics

Main visualizable metrics:

- `mean_team_reward`
- `order_completion_rate`
- `average_profit_per_vehicle`
- `empty_travel_ratio`
- `battery_safety_rate`
- `charging_efficiency`
- `actor_loss`
- `value_loss`
- `entropy`
- `aux_loss`

Generate dashboards:

```powershell
visualize_results --train-dir outputs\comet_run --eval-dirs outputs\eval_test outputs\greedy_test --labels COMET Greedy --output-dir outputs\viz
```

This writes:

- `training_dashboard.png`
- `loss_dashboard.png`
- `validation_dashboard.png` when eval columns exist in training metrics
- `comparison_dashboard.png`
- `comparison_metrics.csv`
- one `*_episode_dashboard.png` per evaluation directory that has `episode_summaries.csv`

## Tests

Run tests without bytecode writes:

```powershell
.\.venv\Scripts\python.exe -B -m pytest
```

## Project Layout

- [`src/comet_taxi`](D:/Codex%20Project/src/comet_taxi): package code
- [`configs`](D:/Codex%20Project/configs): experiment configs
- [`tests`](D:/Codex%20Project/tests): preprocessing, env, model, and smoke training tests
