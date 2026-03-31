# COMET-v2 Taxi：基于真实 NYC TLC 数据驱动的多智能体电动出租车调度原型

## 1. 项目简介

本项目是一个 **COMET-v2 风格** 的多智能体强化学习（MARL）电动出租车调度原型，核心目标是：

- 使用 **真实 NYC TLC Yellow Taxi** 月度订单数据驱动仿真，而不是只用静态统计表或手工合成环境。
- 在同一套环境中保留 **shared actor、ghost + mask、centralized critic、planner / uncertainty gate / fallback** 这些 COMET 系统思想。
- 支持并方便验证 **训练时 10 个智能体、测试时 15 个智能体** 的数量泛化实验。
- 让项目能直接部署到 **Linux + A100** 服务器上跑训练、评估和可视化。

当前仓库默认走 **COMET-v2** 路径，但仍保留 legacy 路径，不会破坏已有 CLI 主入口。

## 2. 模型结构概述

### 2.1 单车局部状态

每个智能体的局部观测至少包含：

- 当前区域（TLC zone，内部编码为 `0..262`，输入网络时用 `zone_id + 1`）
- 电量挡位 one-hot
- 连续电量比例 `battery_ratio`
- 当前模式（idle / serving / charging / repositioning）
- 剩余服务步数

### 2.2 车队编码器

项目不再把手工 `fleet_signature` 当成唯一主表达，而是把它降级为 side / skip feature。现在主车队表示来自：

1. `VehicleTokenEncoder`
   把每辆车的局部状态编码成 token。

2. `FleetSetEncoder`
   把所有真实智能体 token 通过 **permutation-invariant** 的集合编码器聚合成 `fleet_context`。

支持两种模式：

- `deepsets`
- `set_transformer`

这样做的动机不是单纯为了让输入维度固定，而是为了让策略学习：

> “任意一辆同类智能体，在给定自身局部状态、车队上下文和最近历史趋势的条件下，应该如何决策。”

这也是当前仓库回答 `10训15测` 数量泛化问题的核心结构支撑。

### 2.3 时间趋势编码器

`TemporalEncoder` 默认支持：

- `gru`
- `transformer`

它编码最近 `K` 步历史信号，包括：

- demand history
- charge price history
- charger occupancy / queue history
- delay / travel-time residual history

环境中维护统一的历史缓冲区，训练、评估和运行时都走同一条 temporal path。

### 2.4 Planner / Uncertainty / Fallback

运行时不再是“policy logits 直接采样作为最终动作”。默认流程是：

1. 生成候选动作
2. 用 actor logits 作为 prior
3. 用 reward critic 打分
4. 用 cost critics 加惩罚
5. 用 critic ensemble variance 作为不确定性 proxy
6. 超过阈值时触发 uncertainty gate
7. 走 `GreedyDispatchPolicy` fallback

评估指标中会持续记录：

- `fallback_rate`
- `uncertainty_trigger_rate`

### 2.5 Ghost + Mask

这里要明确区分三个概念：

- `nmax`：张量上限，只是为了 batch 化
- `active_agents`：当前回合真实在线车辆数
- `agent_mask`：哪些 slot 是真实车辆，哪些 slot 是 ghost

当前代码中以下路径都显式使用 `agent_mask`：

- actor
- fleet encoder
- critic
- planner
- action masking
- loss 计算
- metric logging

因此：

- 训练时可以设置 `nmax = 16, active_agents = 10`
- 评估时可以设置 `nmax = 16, active_agents = 15`
- 输入张量 shape 保持兼容，但真实有效智能体数量发生变化

## 3. 为什么要用真实 NYC TLC 数据驱动仿真

当前环境不再只依赖静态 demand summary，而是直接使用 **真实 trip 级订单** 作为需求引擎：

- 每天随机抽取真实日期
- 以 `tpep_pickup_datetime` 为基准
- 每 10 分钟推进一步
- 从当前 `time_bin` 的真实订单池中分配订单
- 使用真实 `trip_distance`、`fare_amount / total_amount`、`PULocationID`、`DOLocationID`

这比“先聚合成一张静态统计表再做简单采样”更适合做导师汇报，因为它能明确回答：

- 真实订单流是怎么进入环境的
- 为什么环境里的收益、距离、时间、充电需求有现实依据
- 为什么 `10训15测` 的结果不是建立在过度理想化的玩具环境上

当前实现对真实 TLC 数据做了这些处理：

- 删除 `dropoff <= pickup` 的异常订单
- 删除 `fare_amount <= 0`
- 删除 `trip_distance <= 0`
- 删除 `PULocationID / DOLocationID` 不在 `1..263` 的记录
- 将 `trip_distance` 转成公里，用于电量消耗
- 保留真实收入 `total_amount`（若缺失则退回 `fare_amount`）

## 4. Linux / A100 服务器安装步骤

以下命令默认你已经进入项目根目录。

### 4.1 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4.2 安装 CUDA 版 PyTorch

如果服务器是 CUDA 12.1：

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

如果你使用别的 CUDA 版本，请按服务器上的 CUDA 版本改 PyTorch 下载源。

### 4.3 安装项目依赖与 CLI

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

### 4.5 检查 GPU 是否可用

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

### 4.6 关于 `device = auto`

配置里默认使用：

```toml
[train]
device = "auto"
```

这意味着：

- 如果检测到 CUDA，就自动使用 GPU
- 在 A100 服务器上通常会自动切到 `cuda`
- 在本地 CPU 环境下也能跑测试和 smoke

### 4.7 推荐后台运行方式

使用 `nohup`：

```bash
nohup train --config configs/real_nyc_2026.toml --data-dir data/processed/nyc_real_2026 --output-dir outputs/comet_v2_run > outputs/comet_v2_run/train.log 2>&1 &
```

查看日志：

```bash
tail -f outputs/comet_v2_run/train.log
```

或者使用 `tmux`：

```bash
tmux new -s comet_train
train --config configs/real_nyc_2026.toml --data-dir data/processed/nyc_real_2026 --output-dir outputs/comet_v2_run
```

### 4.8 终端粘贴提醒

从网页复制命令到终端时，不要把 bracketed paste 控制字符一起粘进去；如果你发现命令前后出现异常控制符，请删掉后重新输入。

## 5. 数据集准备

本项目默认使用 **NYC TLC Yellow Taxi** 数据。

### 5.1 推荐文件名

```bash
yellow_tripdata_2026-01.parquet
```

### 5.2 原始数据放置目录

请把数据放到：

```bash
data/raw/yellow_tripdata_2026-01.parquet
```

推荐先建目录：

```bash
mkdir -p data/raw data/processed outputs
ls data/raw/yellow_tripdata_2026-01.parquet
```

### 5.3 预处理输出目录示例

真实数据的预处理输出建议放到：

```bash
data/processed/nyc_real_2026
```

`prepare_nyc` 会输出：

- `train.parquet`
- `val.parquet`
- `test.parquet`
- `metadata.json`

其中：

- `train/val/test.parquet` 是 **订单级** 数据，不再只是静态 demand 汇总表
- `metadata.json` 保存 demand summary、OD summary、trip distance / duration / fare summary、zone neighbors、charge station 初始化信息等

## 6. 数据预处理步骤

### 6.1 常规真实数据预处理

```bash
prepare_nyc --config configs/real_nyc_2026.toml --input data/raw/yellow_tripdata_2026-01.parquet --output data/processed/nyc_real_2026
```

### 6.2 `10训15测` 专用预处理

```bash
prepare_nyc --config configs/ten_to_fifteen_real.toml --input data/raw/yellow_tripdata_2026-01.parquet --output data/processed/ten15_real
```

## 7. Smoke Test 指令

如果你想先确认链路是通的，可以先跑 smoke：

```bash
prepare_nyc --config configs/smoke.toml --input data/raw/yellow_tripdata_2026-01.parquet --output data/processed/smoke_real
train --config configs/smoke.toml --data-dir data/processed/smoke_real --output-dir outputs/smoke_real_run
```

这条路径适合：

- 验证安装是否成功
- 验证 CLI 是否可用
- 验证 trainer / evaluation / checkpoint 是否工作

## 8. 常规训练步骤

### 8.1 预处理

```bash
prepare_nyc --config configs/real_nyc_2026.toml --input data/raw/yellow_tripdata_2026-01.parquet --output data/processed/nyc_real_2026
```

### 8.2 训练

```bash
train --config configs/real_nyc_2026.toml --data-dir data/processed/nyc_real_2026 --output-dir outputs/comet_v2_run
```

训练输出目录示例：

```bash
outputs/comet_v2_run
```

其中通常会包含：

- `metrics.csv`
- `reward_curve.csv`
- `reward_curve.png`
- `offline_dataset.npz`
- `checkpoints/episode_XXXX.pt`
- `config_snapshot.json`

## 9. 常规评估步骤

假设最新 checkpoint 是 `outputs/comet_v2_run/checkpoints/episode_0080.pt`：

```bash
evaluate --config configs/real_nyc_2026.toml --data-dir data/processed/nyc_real_2026 --checkpoint outputs/comet_v2_run/checkpoints/episode_0080.pt --output-dir outputs/eval_standard --split test --episodes 1
```

评估输出目录示例：

```bash
outputs/eval_standard
```

通常包含：

- `metrics.csv`
- `episode_summaries.csv`

## 10. `10训15测` 实验步骤

这条实验的目标是：

- 训练时固定 `10` 个真实智能体
- 测试 stress 场景里使用 `15` 个真实智能体
- 观察策略是否还能在更大 agent 数环境中保持有效性

### 10.1 预处理

```bash
prepare_nyc --config configs/ten_to_fifteen_real.toml --input data/raw/yellow_tripdata_2026-01.parquet --output data/processed/ten15_real
```

### 10.2 训练

```bash
train --config configs/ten_to_fifteen_real.toml --data-dir data/processed/ten15_real --output-dir outputs/ten15_real
```

### 10.3 Stress 评估

把 checkpoint 文件名换成你实际训练出来的最新文件，例如：

```bash
evaluate --config configs/ten_to_fifteen_real.toml --data-dir data/processed/ten15_real --checkpoint outputs/ten15_real/checkpoints/episode_0040.pt --output-dir outputs/ten15_eval --split test --episodes 1 --stress
```

### 10.4 如何确认 `15` 智能体场景真的跑了

查看：

```bash
python - <<'PY'
import pandas as pd
frame = pd.read_csv('outputs/ten15_eval/metrics.csv')
print(frame[['scenario', 'mean_team_reward', 'order_completion_rate']])
PY
```

你应该能在 `scenario` 列中看到：

```text
unseen_fleet_15
```

## 11. Stress 评估步骤

如果你想做更全面的鲁棒性评估，可以直接跑 stress preset：

```bash
evaluate --config configs/stress_eval.toml --data-dir data/processed/nyc_real_2026 --checkpoint outputs/comet_v2_run/checkpoints/episode_0080.pt --output-dir outputs/eval_stress --split test --episodes 1 --stress
```

`metrics.csv` 中至少会包含：

- `mean_team_reward`
- `order_completion_rate`
- `average_profit_per_vehicle`
- `battery_violation_rate`
- `charger_overflow_rate`
- `service_violation_rate`
- `fallback_rate`
- `uncertainty_trigger_rate`
- `scenario`

典型 `scenario` 包括：

- `standard_test`
- `unseen_fleet_15` 或其他 unseen fleet size
- `charger_outage_xx`
- `demand_shock_xx`
- `travel_time_xx`
- `mixed_ood_stress`

## 12. Greedy Baseline

运行启发式基线：

```bash
run_greedy_baseline --config configs/real_nyc_2026.toml --data-dir data/processed/nyc_real_2026 --output-dir outputs/greedy_eval --split test --episodes 1
```

运行 `10训15测` 下的 greedy stress baseline：

```bash
run_greedy_baseline --config configs/ten_to_fifteen_real.toml --data-dir data/processed/ten15_real --output-dir outputs/ten15_greedy --split test --episodes 1 --stress
```

## 13. 可视化命令

如果你已经有训练目录和评估目录，可以执行：

```bash
visualize_results --train-dir outputs/comet_v2_run --eval-dirs outputs/eval_standard outputs/eval_stress outputs/greedy_eval --labels COMET-v2 Stress Greedy --output-dir outputs/viz
```

通常会生成：

- `training_dashboard.png`
- `loss_dashboard.png`
- `validation_dashboard.png`
- `comparison_dashboard.png`
- `robustness_dashboard.png`
- `constraint_dashboard.png`
- `fallback_dashboard.png`
- `comparison_metrics.csv`

## 14. 结果文件在哪里看

推荐重点看这些文件：

- 训练指标：`outputs/comet_v2_run/metrics.csv`
- 训练曲线：`outputs/comet_v2_run/reward_curve.png`
- checkpoint：`outputs/comet_v2_run/checkpoints/*.pt`
- 标准评估：`outputs/eval_standard/metrics.csv`
- stress 评估：`outputs/eval_stress/metrics.csv`
- `10训15测`：`outputs/ten15_eval/metrics.csv`
- 可视化：`outputs/viz/*.png`

## 15. 常见报错排查

### 15.1 `prepare_nyc: command not found`

说明当前环境还没有安装项目入口。执行：

```bash
python -m pip install -e .
which prepare_nyc
```

### 15.2 `train: command not found` 或 `evaluate: command not found`

同样先执行：

```bash
python -m pip install -e .
which train
which evaluate
```

### 15.3 `torch.cuda.is_available() = False`

说明当前虚拟环境里装的可能不是 CUDA 版 PyTorch，或者服务器没有正确暴露 GPU。请先检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

必要时重新安装 CUDA 版 torch。

### 15.4 `TOMLDecodeError` / 配置读取失败

如果你手工编辑过 `.toml`，请确认文件是有效 TOML，且没有混入奇怪字符。

### 15.5 `which prepare_nyc` 没有输出

这通常表示：

- 没激活 `.venv`
- 或者没执行 `pip install -e .`

按下面顺序重试：

```bash
source .venv/bin/activate
python -m pip install -e .
which prepare_nyc
```

## 16. 当前简化点 / 已知局限

当前实现已经可以运行真实 TLC 驱动的 COMET-v2 原型，但仍有一些工程上的简化：

- TLC 数据没有真实电价，因此充电价格使用 **time-of-day proxy**。
- reposition 动作不是基于真实道路网络，而是基于 **历史 OD 转移构造的四邻域 zone graph**。
- 环境中的充电站仍是 **zone-level 近似**，不是精确站点拓扑。
- offline dataset 仍是由 simulator + behavior policy 导出的初始离线数据，不是真实车队日志。
- planner 采用的是轻量候选动作打分器，不是完整 MPC。

不过这些简化都已经在代码结构上留好了继续增强的接口，适合作为导师汇报版本和后续论文 / 项目迭代基础。
