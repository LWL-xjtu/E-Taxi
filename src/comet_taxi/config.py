from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    charger_capacity: int = 4
    max_queue_length: int = 8
    battery_low_threshold: float = 0.25
    battery_mid_threshold: float = 0.60
    charge_rate_per_step: float = 0.18
    move_consumption: float = 0.03
    service_consumption: float = 0.02
    idle_consumption: float = 0.005
    low_battery_penalty_weight: float = 0.35
    empty_move_penalty: float = 0.08
    idle_penalty: float = 0.02
    service_violation_penalty: float = 0.05
    default_charge_price: float = 1.0
    travel_time_noise_base: float = 0.0


@dataclass(slots=True)
class SetEncoderConfig:
    type: str = "deepsets"
    hidden_dim: int = 128
    num_heads: int = 4
    num_inducing_points: int = 8
    pooling: str = "mean"
    use_fleet_signature_skip: bool = True


@dataclass(slots=True)
class TemporalConfig:
    history_len: int = 6
    encoder_type: str = "gru"
    hidden_dim: int = 96
    num_layers: int = 1


@dataclass(slots=True)
class ModelConfig:
    variant: str = "v2"
    zone_embedding_dim: int = 16
    hidden_dim: int = 128
    fleet_hidden_dim: int = 128
    dropout: float = 0.1
    aux_hidden_dim: int = 64
    critic_ensemble_size: int = 2
    vehicle_token_dim: int = 128
    use_layer_norm: bool = True


@dataclass(slots=True)
class TrainConfig:
    seed: int = 7
    device: str = "auto"
    total_episodes: int = 60
    rollout_episodes_per_update: int = 1
    ppo_epochs: int = 4
    minibatch_size: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    aux_coef: float = 0.2
    cost_value_coef: float = 0.3
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
    charger_capacity_scale_min: float = 0.75
    charger_capacity_scale_max: float = 1.0
    peak_shock_probability: float = 0.15
    peak_shock_scale_min: float = 1.1
    peak_shock_scale_max: float = 1.5
    event_day_probability: float = 0.1
    event_day_scale_min: float = 1.2
    event_day_scale_max: float = 1.7
    apply_to_eval: bool = False


@dataclass(slots=True)
class RewardConfig:
    profit_weight: float = 1.0
    efficiency_weight: float = 1.0
    battery_weight: float = 1.0


@dataclass(slots=True)
class OfflineRLConfig:
    enabled: bool = True
    algorithm: str = "iql"
    dataset_episodes: int = 8
    pretrain_epochs: int = 2
    batch_size: int = 32
    expectile: float = 0.7
    temperature: float = 2.0
    cql_alpha: float = 0.1
    awac_lambda: float = 1.0


@dataclass(slots=True)
class OnlineFineTuneConfig:
    offline_ratio_start: float = 0.7
    offline_ratio_end: float = 0.1
    offline_ratio_decay_steps: int = 100
    planner_enabled: bool = True
    uncertainty_threshold: float = 0.25
    online_replay_capacity: int = 4096


@dataclass(slots=True)
class SafetyConfig:
    battery_limit: float = 0.05
    charger_limit: float = 0.10
    service_limit: float = 0.25
    multiplier_init: float = 0.1
    multiplier_lr: float = 0.05
    pid_kp: float = 1.0
    pid_ki: float = 0.1
    pid_kd: float = 0.0


@dataclass(slots=True)
class PlannerConfig:
    top_k_zones: int = 4
    use_value_scoring: bool = True
    use_short_horizon_rollout: bool = False
    fallback_policy: str = "greedy"
    risk_trigger_soc: float = 0.18
    planner_mode: str = "planner"


@dataclass(slots=True)
class EvalConfig:
    standard_episodes: int = 1
    stress_episodes: int = 1
    unseen_fleet_sizes: list[int] = field(default_factory=lambda: [16, 32, 48, 64])
    charger_outage_levels: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5])
    demand_shock_levels: list[float] = field(default_factory=lambda: [1.0, 1.25, 1.5])
    travel_time_shock_levels: list[float] = field(default_factory=lambda: [1.0, 1.15, 1.3])
    event_day_presets: list[str] = field(default_factory=lambda: ["baseline", "concert", "rain"])


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig
    env: EnvConfig
    model: ModelConfig
    set_encoder: SetEncoderConfig
    temporal: TemporalConfig
    train: TrainConfig
    domain_randomization: DomainRandomizationConfig
    reward: RewardConfig
    offline_rl: OfflineRLConfig
    online_finetune: OnlineFineTuneConfig
    safety: SafetyConfig
    planner: PlannerConfig
    evaluation: EvalConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_section(raw: dict[str, Any], key: str, cls: type[Any]) -> Any:
    return cls(**raw.get(key, {}))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return ExperimentConfig(
        data=_load_section(raw, "data", DataConfig),
        env=_load_section(raw, "env", EnvConfig),
        model=_load_section(raw, "model", ModelConfig),
        set_encoder=_load_section(raw, "set_encoder", SetEncoderConfig),
        temporal=_load_section(raw, "temporal", TemporalConfig),
        train=_load_section(raw, "train", TrainConfig),
        domain_randomization=_load_section(
            raw, "domain_randomization", DomainRandomizationConfig
        ),
        reward=_load_section(raw, "reward", RewardConfig),
        offline_rl=_load_section(raw, "offline_rl", OfflineRLConfig),
        online_finetune=_load_section(raw, "online_finetune", OnlineFineTuneConfig),
        safety=_load_section(raw, "safety", SafetyConfig),
        planner=_load_section(raw, "planner", PlannerConfig),
        evaluation=_load_section(raw, "evaluation", EvalConfig),
    )
