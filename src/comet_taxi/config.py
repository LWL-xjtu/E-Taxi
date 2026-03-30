from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass(slots=True)
class DataConfig:
    grid_rows: int = 8
    grid_cols: int = 8
    step_minutes: int = 10
    start_hour: int = 8
    end_hour: int = 20
    use_first_n_days: int = 7
    train_days: int = 5
    val_days: int = 1
    test_days: int = 1

    @property
    def cell_count(self) -> int:
        return self.grid_rows * self.grid_cols

    @property
    def episode_steps(self) -> int:
        minutes = (self.end_hour - self.start_hour) * 60
        return minutes // self.step_minutes


@dataclass(slots=True)
class EnvConfig:
    nmax: int = 64
    min_active_agents: int = 24
    max_active_agents: int = 64
    charge_station_count: int = 5
    battery_low_threshold: float = 0.25
    battery_mid_threshold: float = 0.60
    charge_rate_per_step: float = 0.18
    move_consumption: float = 0.03
    service_consumption: float = 0.02
    idle_consumption: float = 0.005
    low_battery_penalty_weight: float = 0.35
    empty_move_penalty: float = 0.08
    idle_penalty: float = 0.02
    default_charge_price: float = 1.0


@dataclass(slots=True)
class ModelConfig:
    zone_embedding_dim: int = 16
    hidden_dim: int = 128
    fleet_hidden_dim: int = 128
    dropout: float = 0.1
    aux_hidden_dim: int = 64


@dataclass(slots=True)
class TrainConfig:
    seed: int = 7
    device: str = "auto"
    total_episodes: int = 60
    ppo_epochs: int = 4
    minibatch_size: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    aux_coef: float = 0.2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 0.8
    eval_interval: int = 10
    save_interval: int = 10


@dataclass(slots=True)
class DomainRandomizationConfig:
    enabled: bool = True
    demand_scale_min: float = 0.85
    demand_scale_max: float = 1.15
    travel_time_noise_min: float = 0.9
    travel_time_noise_max: float = 1.15
    charge_price_scale_min: float = 0.9
    charge_price_scale_max: float = 1.2


@dataclass(slots=True)
class RewardConfig:
    profit_weight: float = 1.0
    efficiency_weight: float = 1.0
    battery_weight: float = 1.0


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig
    env: EnvConfig
    model: ModelConfig
    train: TrainConfig
    domain_randomization: DomainRandomizationConfig
    reward: RewardConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_section(raw: dict[str, Any], key: str, cls: type[Any]) -> Any:
    section = raw.get(key, {})
    return cls(**section)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return ExperimentConfig(
        data=_load_section(raw, "data", DataConfig),
        env=_load_section(raw, "env", EnvConfig),
        model=_load_section(raw, "model", ModelConfig),
        train=_load_section(raw, "train", TrainConfig),
        domain_randomization=_load_section(
            raw, "domain_randomization", DomainRandomizationConfig
        ),
        reward=_load_section(raw, "reward", RewardConfig),
    )
