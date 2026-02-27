# MADDPG 多智能体路径规划

基于 MADDPG（Multi-Agent Deep Deterministic Policy Gradient）的多智能体路径规划系统。3 个智能体在包含高程、坡度、POI 的公园环境中协同规划路径。

## 目录结构

```
rl1/                                   # 项目根目录
├── maddpg/                            # 核心代码包
│   ├── envs/                          # 环境
│   │   ├── park_env.py                # ParkEnv (64×64 地图, 3 智能体)
│   │   └── wrapper.py                 # MADDPGWrapper (dict→list 转换)
│   ├── models/                        # 模型变体
│   │   ├── hyper_model.py             # CNN-MLP (提出方法)
│   │   ├── mlp_model.py              # 纯 MLP
│   │   ├── unet_model.py             # UNet-MLP
│   │   └── attention_model.py        # Attention-MLP
│   ├── agents/                        # 智能体
│   │   └── simple_agent.py            # MAAgent (CTDE)
│   ├── viz/                           # 可视化工具
│   │   └── nature_style.py            # Nature 期刊配色
│   ├── eval_utils.py                  # 评估工具函数
│   ├── scripts/                       # 训练脚本
│   │   ├── train.py                   # 标准训练
│   │   ├── train_ablation.py          # 消融训练 (模型/环境开关)
│   │   ├── train_with_seed.py         # 多种子训练
│   │   ├── train_independent_ddpg.py  # 独立 DDPG 基线
│   │   └── eval_ablation.py           # 消融评估
│   └── experiments/                   # 实验脚本
│       ├── slope_ablation.py          # 坡度消融可视化
│       ├── reward_analysis.py         # 奖励消融可视化
│       ├── degradation.py             # 退化行为分析
│       ├── multiagent.py              # MADDPG vs 独立 DDPG
│       ├── cooperation.py             # 协作可视化
│       ├── architecture_comparison.py # 架构对比
│       ├── input_ablation.py          # 输入通道消融
│       ├── multi_seed.py              # 多种子统计
│       ├── generalization.py          # 泛化测试
│       ├── zero_shot.py               # 零样本迁移
│       ├── trajectory.py              # 轨迹后处理
│       └── explainability.py          # 可解释性分析
├── data/                              # 数据目录
│   ├── checkpoints/                   # 模型检查点
│   │   ├── baseline/                  # 主 CNN-MLP 模型
│   │   ├── seed0_full/               # 消融 baseline
│   │   ├── A1_nopenalty_seed0/       # 无坡度惩罚
│   │   ├── A2_noobs_seed0/           # 无高程观测
│   │   ├── no_poi_reward/            # 无 POI 奖励
│   │   └── no_trajectory_penalty/    # 无轨迹惩罚
│   ├── training_csvs/                 # 训练曲线 CSV
│   └── results/                       # 实验结果 (图片 + CSV)
├── tests/                             # 测试套件 (68 个测试)
├── parl/                              # PARL 框架 (editable install)
└── setup.py                           # 包安装
```

## 技术栈

| 组件 | 版本 |
|------|------|
| Python | 3.9 |
| PaddlePaddle-GPU | 2.5.1 (CUDA 11.8 Runtime) |
| PARL | 2.2.1 (editable install) |
| gym | 0.26.2 |
| GPU | NVIDIA H20 (Compute 9.0) |
| cuDNN | 9.x (通过 symlink 兼容) |

---

## 快速开始（本地运行）

### 1. 激活环境

```bash
source /mnt/volumes/infra-cloud-alg-sh02/ruihang/venv_rl1/bin/activate
```

激活后会自动设置 `LD_LIBRARY_PATH` 指向 cuDNN 库。

### 2. 验证环境

```bash
# 检查 GPU + cuDNN
python -c "import paddle; paddle.set_device('gpu'); print(paddle.randn([2,3]))"

# 检查模型加载
python -c "
from maddpg.eval_utils import setup_env, build_agents, restore_agents
env = setup_env()
agents = build_agents(env)
restore_agents(agents, 'data/checkpoints/baseline')
obs_n = env.reset()
actions = [a.predict(o) for a, o in zip(agents, obs_n)]
print('OK, actions:', [a.shape for a in actions])
"

# 运行测试套件
pytest tests/ -v
```

### 3. 训练

所有命令从项目根目录 `rl1/` 执行。

**标准训练（CNN-MLP 模型，100 万 episode）：**

```bash
python -m maddpg.scripts.train \
    --max_episodes 1000000 \
    --model_dir data/checkpoints/baseline
```

**消融训练：**

```bash
# 完整 baseline
python -m maddpg.scripts.train_ablation \
    --max_episodes 1000000 \
    --model_dir data/checkpoints/seed0_full

# 无坡度惩罚
python -m maddpg.scripts.train_ablation \
    --no_slope_penalty \
    --max_episodes 1000000 \
    --model_dir data/checkpoints/A1_nopenalty_seed0

# 无高程观测
python -m maddpg.scripts.train_ablation \
    --no_elevation_obs \
    --max_episodes 1000000 \
    --model_dir data/checkpoints/A2_noobs_seed0

# 平坦地形
python -m maddpg.scripts.train_ablation \
    --flat_terrain \
    --max_episodes 1000000 \
    --model_dir data/checkpoints/flat

# 无 POI 奖励
python -m maddpg.scripts.train_ablation \
    --no_poi_reward \
    --max_episodes 1000000 \
    --model_dir data/checkpoints/no_poi_reward

# 无轨迹惩罚
python -m maddpg.scripts.train_ablation \
    --no_trajectory_penalty \
    --max_episodes 1000000 \
    --model_dir data/checkpoints/no_trajectory_penalty

# 使用纯 MLP / UNet / Attention 模型
python -m maddpg.scripts.train_ablation \
    --model_type mlp \
    --max_episodes 1000000 \
    --model_dir data/checkpoints/mlp_model
```

**多种子训练：**

```bash
python -m maddpg.scripts.train_with_seed --seed 0 --model_dir data/checkpoints/seed_0
python -m maddpg.scripts.train_with_seed --seed 1 --model_dir data/checkpoints/seed_1
python -m maddpg.scripts.train_with_seed --seed 2 --model_dir data/checkpoints/seed_2
```

**独立 DDPG 基线：**

```bash
python -m maddpg.scripts.train_independent_ddpg \
    --max_episodes 1000000 \
    --model_dir data/checkpoints/independent_ddpg
```

**从断点恢复：**

```bash
python -m maddpg.scripts.train \
    --restore \
    --model_dir data/checkpoints/baseline
```

长时间训练建议使用 `tmux` 或 `screen`：

```bash
tmux new -s train
source /mnt/volumes/infra-cloud-alg-sh02/ruihang/venv_rl1/bin/activate
python -m maddpg.scripts.train_ablation --max_episodes 1000000 --model_dir data/checkpoints/seed0_full
# Ctrl+B D 退出 tmux，训练继续运行
```

### 4. 评估与实验

```bash
# 模型评估
python -m maddpg.scripts.eval_ablation \
    --model_dir data/checkpoints/baseline \
    --config seed0_full \
    --episodes 100

# 坡度消融可视化（读取训练 CSV）
python -m maddpg.experiments.slope_ablation \
    --csv_dir data/training_csvs \
    --output_dir data/results/slope_ablation

# 奖励消融可视化（读取训练 CSV）
python -m maddpg.experiments.reward_analysis \
    --csv_dir data/training_csvs \
    --output_dir data/results/reward_analysis

# 退化行为分析
python -m maddpg.experiments.degradation \
    --model_dir data/checkpoints/baseline \
    --output_dir data/results/degradation

# MADDPG vs 独立 DDPG
python -m maddpg.experiments.multiagent \
    --maddpg_dir data/checkpoints/baseline \
    --output_dir data/results/multiagent

# 协作可视化
python -m maddpg.experiments.cooperation \
    --model_dir data/checkpoints/baseline \
    --output_dir data/results/cooperation

# 输入通道消融
python -m maddpg.experiments.input_ablation \
    --model_dir data/checkpoints/baseline \
    --output_dir data/results/input_ablation

# 泛化测试
python -m maddpg.experiments.generalization \
    --model_dir data/checkpoints/baseline \
    --output_dir data/results/generalization

# 零样本迁移
python -m maddpg.experiments.zero_shot \
    --model_dir data/checkpoints/baseline \
    --output_dir data/results/zero_shot

# 轨迹后处理
python -m maddpg.experiments.trajectory \
    --model_dir data/checkpoints/baseline \
    --output_dir data/results/trajectory

# 可解释性分析
python -m maddpg.experiments.explainability \
    --model_dir data/checkpoints/baseline \
    --output_dir data/results/explainability
```

### 5. 运行测试

```bash
pytest tests/ -v
```

---

## Docker 使用（可选）

如需在 Docker 中运行，项目提供了 `maddpg/Dockerfile`。

```bash
cd maddpg/
docker build -t maddpg:latest .

# 交互式
docker run -it --gpus all \
    -v $(pwd)/../data:/data \
    --shm-size=4g \
    maddpg:latest bash

# 后台训练
docker run -d --gpus all \
    -v $(pwd)/../data:/data \
    --shm-size=4g \
    --restart unless-stopped \
    maddpg:latest \
    python -m maddpg.scripts.train_ablation \
        --max_episodes 1000000 \
        --model_dir /data/checkpoints/seed0_full
```

---

## 关键超参数

| 参数 | 值 | 说明 |
|------|----|------|
| `CRITIC_LR` | 0.001 | Critic 学习率 |
| `ACTOR_LR` | 0.0001 | Actor 学习率 |
| `GAMMA` | 0.95 | 折扣因子 |
| `TAU` | 0.001 | 软更新系数 |
| `BATCH_SIZE` | 1024 | 批量大小 |
| `MAX_STEP_PER_EPISODE` | 120 | 每 episode 最大步数 |
| 回放缓冲区 | 1,000,000 | 经验回放大小 |
| 梯度裁剪 | max_norm=0.5 | Actor/Critic |

## 模型架构

**CTDE（集中训练、分散执行）：**
- 训练: Critic 输入全部 3 个智能体的观测+动作 (180x3 + 4x3 = 552 维)
- 执行: Actor 仅用自身观测 (180 维)

**观测结构 (180 维)：**
- `[0:59]` 位置、相对位置、POI -> MLP 分支
- `[59:180]` 11x11 局部高程图 -> CNN/UNet/Attention 分支

**4 种模型变体：**

| 模型 | 高程处理 | 用途 |
|------|---------|------|
| CNN-MLP | Conv2D(1->16->32) + MaxPool | 提出方法 |
| Pure MLP | 直接拼接 180 维 | 消融对比 |
| UNet-MLP | UNet 编码解码 + 跳跃连接 | 消融对比 |
| Attention-MLP | 自注意力 (4 头) + 池化 | 消融对比 |

## 消融环境开关

| 参数 | 默认 | 效果 |
|------|------|------|
| `--no_elevation_obs` | 启用 | 移除 121 维高程观测 |
| `--no_slope_penalty` | 启用 | 移除陡坡惩罚 (-5) |
| `--flat_terrain` | 关闭 | 地形全部置零 |
| `--no_poi_reward` | 启用 | 移除 POI 访问奖励 |
| `--no_trajectory_penalty` | 启用 | 移除轨迹交叉/卡住惩罚 |

## 已有检查点

| 配置 | 路径 | 说明 |
|------|------|------|
| CNN-MLP baseline | `data/checkpoints/baseline/` | 主模型 |
| 消融 baseline | `data/checkpoints/seed0_full/` | 完整环境 |
| 无坡度惩罚 | `data/checkpoints/A1_nopenalty_seed0/` | 移除坡度惩罚 |
| 无高程观测 | `data/checkpoints/A2_noobs_seed0/` | 移除高程观测 |
| 无 POI 奖励 | `data/checkpoints/no_poi_reward/` | 移除 POI 奖励 |
| 无轨迹惩罚 | `data/checkpoints/no_trajectory_penalty/` | 移除轨迹惩罚 |

每个检查点目录包含 `agent_0`、`agent_1`、`agent_2` 三个 PaddlePaddle 参数文件。

## License

Apache License 2.0
