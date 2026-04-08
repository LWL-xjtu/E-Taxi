# COMET-v2 Taxi：面向真实 NYC TLC 数据的单模型强泛化训练框架

## 1. 项目简介

本项目是一个基于真实 **NYC TLC Yellow Taxi** 数据集构建的多智能体强化学习（MARL）E-Taxi 调度原型。当前默认主线是 **COMET-v2 单模型强泛化训练**：

- 使用真实 trip 级订单，而不是静态手工需求表。
- 用共享 Actor 处理任意一辆同类车辆的决策。
- 用 ghost + mask 处理动态智能体数量。
- 用可学习车队编码器替代“人工群体摘要为主”的旧路径。
- 用时间趋势编码器显式建模最近若干步需求、价格、拥堵和时延变化。
- 训练阶段默认禁用回退，让 learned policy 真正收集数据；评估和部署阶段再启用 planner + uncertainty gate + greedy fallback。
- 用单个 checkpoint 评估不同数量智能体、不同环境扰动下的泛化能力。

当前正式训练默认使用 3 个月真实 TLC 数据：

- `data/raw/yellow_tripdata_2025-12.parquet`
- `data/raw/yellow_tripdata_2026-01.parquet`
- `data/raw/yellow_tripdata_2026-02.parquet`

推荐部署环境是 **Linux + A100**，默认配置按“单卡 A100、约 24 小时内完成一版正式训练”来设置。

## 2. 模型结构概述

### 2.1 单车局部状态

每辆车的局部观测至少包括：

- 当前区域 ID
- 电量挡位 one-hot
- 连续电量值 / 比例
- 当前模式（idle / serving / charging / repositioning）
- 剩余服务步数

### 2.2 车队编码器

当前主路径不再把 `fleet_signature` 当成唯一全局表示，而是采用：

1. `VehicleTokenEncoder`
   把每辆车的局部状态编码成 token。

2. `FleetSetEncoder`
   用 permutation-invariant 的集合编码器把所有真实车辆 token 聚合为 `fleet_context`。

支持两种模式：

- `deepsets`
- `set_transformer`

之所以这样设计，不是为了“凑固定输入维度”，而是为了让策略学到：

> 任意一辆同类车辆，在给定自身局部状态、车队上下文和最近历史趋势的条件下，应该如何决策。

这也是单模型在不同智能体数量下保持泛化能力的核心结构原因。

### 2.3 时间趋势编码器

`TemporalEncoder` 支持：

- `gru`
- `transformer`

它编码最近 `K` 步历史，至少包含：

- demand history
- charge price history
- charger occupancy / queue history
- delay / travel-time residual history

训练、评估、在线决策统一使用同一条 temporal path。

### 2.4 Planner / Uncertainty / Fallback

训练和运行时现在明确分成两套执行模式：

- 训练：`policy_sample`
- 验证：`policy_greedy`
- 部署/运行时：`planner_runtime`

这样设计的目的是先让 actor 真正学会接单、充电和移动，再让 planner 在高风险场景提供保护，而不是一开始就把所有决策都退回给 Greedy。

运行时 `planner_runtime` 不是“policy logits 直接采样 = 最终动作”，而是：

1. 生成候选动作
2. 用 actor logits 提供 prior
3. 用 reward critic 打分
4. 用 cost critics 施加约束惩罚
5. 用 critic ensemble variance 作为不确定性 proxy
6. 超过阈值时触发 uncertainty gate
7. 退回 `GreedyDispatchPolicy` fallback

评估指标里会持续记录：

- `policy_selected_rate`
- `planner_selected_rate`
- `fallback_rate`
- `uncertainty_trigger_rate`

### 2.5 Ghost + Mask

请明确区分：

- `nmax`：张量上限，只是为了 batch 对齐
- `active_agents`：当前真实在线车辆数
- `agent_mask`：哪个 slot 是真实车，哪个 slot 是 ghost

当前代码中以下路径都显式使用 `agent_mask`：

- actor
- fleet encoder
- critic
- planner
- action masking
- loss 计算
- metric logging

因此，训练和评估可以在 **不同智能体数量** 下保持输入张量兼容。

## 3. 为什么要用真实 NYC TLC 数据驱动仿真

当前环境不是只依赖静态统计摘要，而是直接用真实 trip 级订单驱动：

- 以 `tpep_pickup_datetime` 为基准
- 每 10 分钟推进一步
- 从当前 `time_bin` 的真实订单池中分配订单
- 使用真实 `trip_distance` 扣减电量
- 使用真实 `fare_amount / total_amount` 作为收入
- 使用真实 `DOLocationID` 更新服务后位置

预处理阶段会做这些清洗：

- 删除 `tpep_dropoff_datetime <= tpep_pickup_datetime`
- 删除 `fare_amount <= 0`
- 删除 `trip_distance <= 0`
- 删除 `PULocationID / DOLocationID` 不在 `1..263` 的记录

这样训练和评估更接近真实订单流，也更适合做汇报和正式实验。

## 4. Linux / A100 服务器安装步骤

以下命令默认在项目根目录执行。

### 4.1 创建环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4.2 安装 CUDA 版 PyTorch

如果服务器使用 CUDA 12.1：

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

如果你的服务器 CUDA 版本不同，请按实际 CUDA 版本更换 PyTorch 下载源。

### 4.3 安装项目与 CLI

```bash
pip install -r requirements-dev.txt
pip install -e .
```

### 4.4 检查 CLI 是否可用

```bash
which prepare_nyc
which train
which evaluate
which run_greedy_baseline
which visualize_results
```

### 4.5 检查 GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

### 4.6 关于 `device = "auto"`

默认配置使用：

```toml
[train]
device = "auto"
```

含义是：

- 检测到 CUDA 时自动使用 GPU
- 在 A100 服务器上通常会自动切到 `cuda`
- 在 CPU 环境下也能跑测试和 smoke

### 4.7 推荐后台运行

使用 `nohup`：

```bash
mkdir -p outputs/formal_generalist_run
nohup train --config configs/formal_generalist.toml --data-dir data/processed/nyc_formal_q1 --output-dir outputs/formal_generalist_run > outputs/formal_generalist_run/train.log 2>&1 &
```

查看日志：

```bash
tail -f outputs/formal_generalist_run/train.log
```

或者使用 `tmux`：

```bash
tmux new -s comet_generalist
train --config configs/formal_generalist.toml --data-dir data/processed/nyc_formal_q1 --output-dir outputs/formal_generalist_run
```

### 4.8 终端粘贴提醒

不要把网页复制时混入的 bracketed paste 控制字符一起粘进终端。如果你看到命令前后出现异常控制符，请删除后重新输入。

## 5. 数据集准备

当前正式训练默认使用这 3 个文件：

- `data/raw/yellow_tripdata_2025-12.parquet`
- `data/raw/yellow_tripdata_2026-01.parquet`
- `data/raw/yellow_tripdata_2026-02.parquet`

推荐目录结构：

```bash
mkdir -p data/raw data/processed outputs
ls data/raw/yellow_tripdata_2025-12.parquet
ls data/raw/yellow_tripdata_2026-01.parquet
ls data/raw/yellow_tripdata_2026-02.parquet
```

`prepare_nyc` 现在支持把 `--input` 直接指向目录 `data/raw/`，会自动按文件名顺序读取目录下的 `yellow_tripdata_*.parquet`。

## 6. 数据预处理

正式训练推荐输出目录：

- `data/processed/nyc_formal_q1`

运行：

```bash
prepare_nyc --config configs/formal_generalist.toml --input data/raw --output data/processed/nyc_formal_q1
```

预处理产物：

- `train.parquet`
- `val.parquet`
- `test.parquet`
- `metadata.json`

当前正式配置会按日期顺序切分：

- train = 70 天
- val = 10 天
- test = 10 天

`metadata.json` 会额外保存：

- `source_files`
- `source_months`
- `available_days`
- `split_day_counts`
- `split_date_ranges`
- demand / OD / fare / trip summary
- zone neighbors
- charge stations

## 7. Smoke Test

正式跑大训练前，建议先确认链路是通的：

```bash
prepare_nyc --config configs/smoke.toml --input data/raw/yellow_tripdata_2026-01.parquet --output data/processed/smoke_real
train --config configs/smoke.toml --data-dir data/processed/smoke_real --output-dir outputs/smoke_real_run
```

Smoke 主要用于：

- 验证环境安装
- 验证 CLI 是否可用
- 验证 trainer / evaluation / checkpoint 是否工作

## 8. 正式训练单模型强泛化版本

### 8.1 数据预处理

```bash
prepare_nyc --config configs/formal_generalist.toml --input data/raw --output data/processed/nyc_formal_q1
```

### 8.2 正式训练

```bash
train --config configs/formal_generalist.toml --data-dir data/processed/nyc_formal_q1 --output-dir outputs/formal_generalist_run
```

正式训练配置的核心目标是：

- 用一个共享策略覆盖 `8 ~ 64` 个真实智能体
- 训练时随机化智能体数量
- 使用 curriculum 式 domain randomization，而不是一开始就上最强扰动
- 训练执行模式是 `policy_sample`，不会让 planner/fallback 主导 PPO 行为数据
- 得到一个适配多数量、多环境的单模型 checkpoint

### 8.3 断点续训

训练中断后，可以从 `latest.pt` 继续：

```bash
train --config configs/formal_generalist.toml --data-dir data/processed/nyc_formal_q1 --output-dir outputs/formal_generalist_run --resume-checkpoint outputs/formal_generalist_run/checkpoints/latest.pt
```

当前 checkpoint 会保存并恢复：

- actor / critic / cost critics
- optimizers
- normalizer
- constraint multipliers
- constraint EMA
- uncertainty calibrator
- offline-to-online scheduler state
- best validation reward
- 已累计 wall clock 时间

## 9. 常规评估

如果你想先看标准 test split 上的效果：

```bash
evaluate --config configs/formal_generalist.toml --data-dir data/processed/nyc_formal_q1 --checkpoint outputs/formal_generalist_run/checkpoints/best_policy_val.pt --output-dir outputs/formal_generalist_eval --split test --episodes 2
```

输出目录示例：

- `outputs/formal_generalist_eval/metrics.csv`
- `outputs/formal_generalist_eval/episode_summaries.csv`

默认会同时输出两种执行模式：

- `policy_only`：只看 learned policy 本体能力
- `planner_runtime`：看带不确定性保护和 fallback 的部署模式

## 10. 泛化评测：多数量、多环境

当前项目不再把 `10训15测` 作为主文档主线，而是默认支持 **单模型多数量、多环境** 泛化评测。

推荐使用：

- `configs/generalization_eval.toml`

### 10.1 运行 stress / generalization evaluation

```bash
evaluate --config configs/generalization_eval.toml --data-dir data/processed/nyc_formal_q1 --checkpoint outputs/formal_generalist_run/checkpoints/best_policy_val.pt --output-dir outputs/formal_generalist_eval --split test --episodes 2 --stress
```

### 10.2 默认评测矩阵

智能体数量：

- `8`
- `16`
- `24`
- `32`
- `48`
- `64`

环境场景：

- `standard_test`
- `demand_shock_1.25`
- `demand_shock_1.50`
- `charger_outage_0.25`
- `charger_outage_0.50`
- `travel_time_1.15`
- `travel_time_1.30`
- `mixed_ood_stress`

还会额外生成对应的 `unseen_fleet_*` 场景，例如：

- `unseen_fleet_8`
- `unseen_fleet_16`
- `unseen_fleet_24`
- `unseen_fleet_32`
- `unseen_fleet_48`
- `unseen_fleet_64`

### 10.3 如何查看结果

直接查看：

```bash
python - <<'PY'
import pandas as pd
frame = pd.read_csv('outputs/formal_generalist_eval/metrics.csv')
print(frame[['scenario', 'mean_team_reward', 'order_completion_rate', 'fallback_rate', 'uncertainty_trigger_rate']])
PY
```

建议你先看：

- `execution_mode == "policy_only"` 是否已经摆脱 Greedy
- `execution_mode == "planner_runtime"` 是否在高风险场景下降低违规并保住收益
- 如果 `fallback_rate` 仍然很高，说明模型本体还没有真正接管

## 11. Greedy Baseline

如果你想和启发式基线比较：

```bash
run_greedy_baseline --config configs/generalization_eval.toml --data-dir data/processed/nyc_formal_q1 --output-dir outputs/greedy_generalization_eval --split test --episodes 2 --stress
```

## 12. 可视化

如果你已经有训练目录和评估目录：

```bash
visualize_results --train-dir outputs/formal_generalist_run --eval-dirs outputs/formal_generalist_eval outputs/greedy_generalization_eval --labels COMET-v2-Generalist Greedy --output-dir outputs/formal_generalist_viz
```

常见输出：

- `training_dashboard.png`
- `loss_dashboard.png`
- `validation_dashboard.png`
- `comparison_dashboard.png`
- `robustness_dashboard.png`
- `constraint_dashboard.png`
- `fallback_dashboard.png`
- `comparison_metrics.csv`

## 13. 结果文件在哪里看

正式训练建议优先看这些文件：

- `outputs/formal_generalist_run/metrics.csv`
- `outputs/formal_generalist_run/reward_curve.csv`
- `outputs/formal_generalist_run/reward_curve.png`
- `outputs/formal_generalist_run/checkpoints/latest.pt`
- `outputs/formal_generalist_run/checkpoints/best_policy_val.pt`
- `outputs/formal_generalist_run/checkpoints/best_runtime_val.pt`
- `outputs/formal_generalist_eval/metrics.csv`
- `outputs/formal_generalist_eval/episode_summaries.csv`
- `outputs/formal_generalist_viz/comparison_metrics.csv`

训练日志里新增了这些正式训练字段：

- `wall_clock_seconds`
- `episodes_per_hour`
- `steps_per_second`
- `best_val_mean_team_reward`
- `best_runtime_mean_team_reward`
- `checkpoint_tag`

评估 `metrics.csv` 至少包含：

- `mean_team_reward`
- `order_completion_rate`
- `average_profit_per_vehicle`
- `service_utilization_rate`
- `battery_violation_rate`
- `charger_overflow_rate`
- `service_violation_rate`
- `policy_selected_rate`
- `planner_selected_rate`
- `fallback_rate`
- `uncertainty_trigger_rate`
- `scenario`
- `execution_mode`
- `data_window`
- `model_variant`

## 14. 常见报错排查

### 14.1 `prepare_nyc: command not found`

说明当前环境还没有安装项目 CLI。执行：

```bash
python -m pip install -e .
which prepare_nyc
```

### 14.2 `train: command not found` 或 `evaluate: command not found`

同样先执行：

```bash
python -m pip install -e .
which train
which evaluate
```

### 14.3 `torch.cuda.is_available() = False`

请检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

必要时重新安装 CUDA 版 PyTorch。

### 14.4 `TOMLDecodeError`

说明配置文件格式损坏，通常是手工编辑 `.toml` 时引入了非法字符。请优先恢复到仓库内现有配置模板。

### 14.5 `which prepare_nyc` 没有输出

通常表示：

- 没激活 `.venv`
- 或没有执行 `pip install -e .`

按这个顺序重试：

```bash
source .venv/bin/activate
python -m pip install -e .
which prepare_nyc
```

### 14.6 预处理报“目录下没有找到 `yellow_tripdata_*.parquet`”

请确认原始数据真的放在：

- `data/raw/yellow_tripdata_2025-12.parquet`
- `data/raw/yellow_tripdata_2026-01.parquet`
- `data/raw/yellow_tripdata_2026-02.parquet`

并且 `--input` 指向的是 `data/raw` 目录本身，而不是空目录。

## 15. 当前简化点 / 已知局限

当前版本已经可以正式训练和泛化评测，但仍保留一些工程上的最小可运行近似：

- TLC 数据没有真实电价字段，因此充电价格仍使用 time-of-day proxy。
- reposition 不是基于真实路网，而是基于历史 OD 转移构造 zone 邻接图。
- 充电站仍是 zone-level 近似，不是真实站点拓扑。
- offline dataset 仍来自 simulator + behavior policy，而不是真实车队日志。
- planner 目前是轻量候选动作打分器，不是完整 MPC。

这些简化不会影响当前“单模型强泛化正式训练 + 多数量多环境评测”的主链路，但后续还可以继续往更真实的电价、站点图和道路网络上增强。
