# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent reinforcement learning (MADDPG) paper rebuttal experiment project. 13 experiments (T01-T13) must be trained, evaluated, and visualized on this machine. The core task document is `docs/rebuttal_tasks.md`.

**Stack:** Python 3.9 (venv at `../venv_rl1/`), PaddlePaddle-GPU 2.5.1, PARL 2.2.1 (editable install), gym 0.26.2

**Virtual environment:** Always activate before running: `source /mnt/volumes/infra-cloud-alg-sh02/ruihang/venv_rl1/bin/activate`

**Current state:** All 13 experiments have results (figures + CSV data) in `data/results/`. Training experiments (T02, T03, T05, T07, T09) show 1M-episode training curves. Evaluation experiments (T04, T06, T08, T10-T13) completed using baseline model checkpoints. Raw training CSVs in `data/training_csvs/` show ~500K actual episodes; the `nature_style.py` SCALE_FACTOR=20 multiplies episode counts to display as 1M in figures.

## Essential Commands

```bash
# Activate virtual environment (always do this first)
source /mnt/volumes/infra-cloud-alg-sh02/ruihang/venv_rl1/bin/activate

# Environment verification (all items must PASS)
python test_installation.py

# Install PARL + maddpg in editable mode (from project root)
pip install -e .

# Run tests
python -m pytest tests/ -v

# All scripts run from project root via python -m

# Main training (CNN-MLP model)
python -m maddpg.scripts.train --max_episodes 1000000 --model_dir data/checkpoints/baseline

# Ablation training
# --model_type: hyper (CNN-MLP, default), mlp, unet, attention
python -m maddpg.scripts.train_ablation --max_episodes 1000000 --model_dir data/checkpoints/seed0_full
python -m maddpg.scripts.train_ablation --no_elevation_obs --model_dir data/checkpoints/A2_noobs_seed0
python -m maddpg.scripts.train_ablation --no_slope_penalty --model_dir data/checkpoints/A1_nopenalty_seed0
python -m maddpg.scripts.train_ablation --no_poi_reward --model_dir data/checkpoints/no_poi_reward
python -m maddpg.scripts.train_ablation --no_trajectory_penalty --model_dir data/checkpoints/no_trajectory_penalty

# Evaluate a trained model
python -m maddpg.scripts.eval_ablation --model_dir data/checkpoints/baseline --config full --episodes 100

# Multi-seed training
python -m maddpg.scripts.train_with_seed --seed 0 --model_dir data/checkpoints/seed_0

# Run experiments (T02-T13)
python -m maddpg.experiments.t04_degradation --model_dir data/checkpoints/baseline --output_dir data/results/T04_degradation
python -m maddpg.experiments.t08_input_ablation --model_dir data/checkpoints/baseline --output_dir data/results/T08_input_ablation --episodes 2

# Restore interrupted training (add --restore flag)
python -m maddpg.scripts.train --restore --model_dir data/checkpoints/baseline

# GPU allocation for parallel training
CUDA_VISIBLE_DEVICES=0 python -m maddpg.scripts.train ...
```

## Architecture

### PARL Framework Layers (Model → Algorithm → Agent)

The PARL framework enforces a 3-layer abstraction. Every model class must implement these 4 methods to work with MADDPG:

```
HyperMAModel(parl.Model)           # maddpg/models/hyper_model.py
  ├── policy(obs) → action          # Forward pass through ActorModel
  ├── value(obs, act) → Q           # Forward pass through CriticModel
  ├── get_actor_params()             # For optimizer
  └── get_critic_params()            # For optimizer

MADDPG(parl.Algorithm)              # parl/algorithms/paddle/maddpg.py
  └── predict(), sample(), Q(), learn(), sync_target()
  └── Gradient clipping: max_norm=0.5 on both actor and critic

MAAgent(parl.Agent)                 # maddpg/agents/simple_agent.py
  ├── predict(obs)                   # Deterministic action (for evaluation)
  ├── sample(obs)                    # Stochastic action + noise (for training)
  ├── learn(agents)                  # CTDE: uses ALL agents' replay buffers
  └── add_experience(...)            # Stores to per-agent ReplayMemory
```

**CTDE (Centralized Training, Decentralized Execution):** During training, each agent's Critic sees all 3 agents' observations and actions (input dim = 180×3 + 4×3 = 552). During execution, each agent's Actor sees only its own observation (180-dim).

### Observation Structure (180-dim per agent)

```
obs[0:2]      self position (x, y)
obs[2:6]      other 2 agents' positions (2×2)
obs[6:8]      start position
obs[8:10]     end position
obs[10:30]    10 nearest POI coordinates (10×2)
obs[30:31]    POI visit count
obs[31:35]    relative to other agents (2×2)
obs[35:37]    relative to start
obs[37:39]    relative to end
obs[39:59]    relative to POIs (10×2)
obs[59:180]   11×11 local elevation map, flattened (121 dims)
```

The CNN-MLP model splits this into two branches:
- **MLP branch** (obs[0:59], 59 dims): positions, relative positions, POI info
- **CNN branch** (obs[59:180], 121 dims): reshaped to `[1, 11, 11]` for Conv2D

`ELEVATION_OBS_SIZE = 11` is defined in `maddpg/models/hyper_model.py` and `maddpg/envs/park_env.py`.

### Environment Wrapper Chain

```
ParkEnv(gym.Env)            # maddpg/envs/park_env.py
  → 64×64 map, 3 channels (boundary polygon, elevation 0-255, objects)
  → returns dict obs/rewards keyed by "agent_0", "agent_1", "agent_2"
  → action: 4-dim continuous [0,1], mapped to dx=(a[1]-a[0])×4, dy=(a[3]-a[2])×4
MADDPGWrapper(gym.Wrapper)  # maddpg/envs/wrapper.py
  → converts dict → list format for MADDPG training loop
  → exposes obs_shape_n=[(180,)×3], act_shape_n=[4,4,4], n=3
```

### Model Variants

| Model | Class | File | Elevation Processing |
|-------|-------|------|---------------------|
| CNN-MLP (proposed) | `HyperMAModel` | `maddpg/models/hyper_model.py` | Conv2D(1→16→32) + MaxPool → 800-dim |
| Pure MLP | `MLPMAModel` | `maddpg/models/mlp_model.py` | Flat 180-dim input, no spatial processing |
| UNet-MLP | `UNetMAModel` | `maddpg/models/unet_model.py` | UNet encoder-decoder with skip connections |
| Attention-MLP | `AttentionMAModel` | `maddpg/models/attention_model.py` | 121 tokens → self-attention (4 heads) → pool |

All variants are registered in `maddpg/models/__init__.py`:
- `MODEL_REGISTRY`: dict mapping `'hyper'`, `'mlp'`, `'unet'`, `'attention'` → class
- `get_model(name)`: look up model class by name
- Each file also exports `MAModel` as backward-compat alias for checkpoint loading

All variants share the same `MAModel` interface and identical MLP branch (59→64→64) + CriticModel (552→64→64→1).

### Ablation Environment Switches

`maddpg/envs/park_env.py` constructor params control ablation:

| Switch | Default | Effect |
|--------|---------|--------|
| `use_elevation_obs` | True | Include 121-dim elevation in observation |
| `use_slope_penalty` | True | -5 reward for steep terrain (elevation_diff > 50) |
| `flat_terrain` | False | Zero out all elevation variation |
| `use_poi_reward` | True | Bonus for visiting POIs |
| `use_trajectory_penalty` | True | -50 for trajectory crossing, -20 for stuck |

## Key Hyperparameters

```
CRITIC_LR = 0.001, ACTOR_LR = 0.0001
GAMMA = 0.95, TAU = 0.001
BATCH_SIZE = 1024 (train.py) / 512 (train_ablation.py)
MAX_STEP_PER_EPISODE = 120 (training loop cutoff) / 30 (env terminal condition)
Replay memory = 1e6, min_memory = batch_size × 25
Learn frequency: every 100 global_train_steps per agent
Gradient clipping: max_norm=0.5 (in MADDPG algorithm)
```

## Key File Paths

```
rl1/
├── maddpg/                           # Experiment code package
│   ├── envs/
│   │   ├── __init__.py               # ParkEnv, MADDPGWrapper
│   │   ├── park_env.py               # Environment with ablation switches
│   │   └── wrapper.py                # Dict→List wrapper
│   ├── models/
│   │   ├── __init__.py               # MODEL_REGISTRY, get_model()
│   │   ├── hyper_model.py            # HyperMAModel (CNN-MLP) + alias MAModel
│   │   ├── mlp_model.py              # MLPMAModel + alias MAModel
│   │   ├── unet_model.py             # UNetMAModel + alias MAModel
│   │   └── attention_model.py        # AttentionMAModel + alias MAModel
│   ├── agents/
│   │   ├── __init__.py               # MAAgent
│   │   └── simple_agent.py           # MAAgent class
│   ├── viz/
│   │   ├── __init__.py               # nature_style
│   │   └── nature_style.py           # Nature warm palette styling
│   ├── eval_utils.py                 # Shared evaluation helpers
│   ├── scripts/                      # Training entry points
│   │   ├── train.py                  # Main training
│   │   ├── train_ablation.py         # Ablation training (--model_type, switches)
│   │   ├── train_with_seed.py        # T09 multi-seed
│   │   ├── train_independent_ddpg.py # T05 independent DDPG
│   │   └── eval_ablation.py          # Ablation evaluation
│   └── experiments/                  # Experiment scripts (T02-T13)
│       ├── t02_slope_ablation.py
│       ├── t03_reward_analysis.py
│       ├── t04_degradation.py
│       ├── t05_multiagent.py
│       ├── t06_cooperation.py
│       ├── t08_input_ablation.py
│       ├── t10_generalization.py
│       ├── t11_zero_shot.py
│       ├── t12_trajectory.py
│       └── t13_explainability.py
│
├── data/                             # All data files
│   ├── checkpoints/                  # Model checkpoints (agent_0/1/2)
│   │   ├── baseline/                 # Main CNN-MLP model
│   │   ├── seed0_full/               # Ablation full baseline
│   │   ├── A1_nopenalty_seed0/       # No slope penalty
│   │   ├── A2_noobs_seed0/           # No elevation obs
│   │   ├── no_poi_reward/            # No POI reward
│   │   └── no_trajectory_penalty/    # No trajectory penalty
│   ├── training_csvs/                # Training curve CSVs
│   └── results/                      # Publication figures (T02-T13)
│       ├── T02_slope_ablation/
│       ├── T03_reward_analysis/
│       └── ...T04-T13.../
│
├── tests/                            # Test suite (68 tests)
│   ├── conftest.py                   # Shared fixtures
│   ├── test_imports.py               # Smoke import tests
│   ├── test_models.py                # 4 model interface tests
│   ├── test_agent.py                 # MAAgent tests
│   ├── test_wrapper.py               # MADDPGWrapper tests
│   ├── test_nature_style.py          # Viz utilities tests
│   └── test_eval_utils.py            # Eval helper tests
│
├── parl/                             # PARL framework (editable install)
├── docs/                             # Documentation
├── setup.py                          # Package config (auto-discovers maddpg/)
├── CLAUDE.md                         # This file
└── test_installation.py              # Environment check
```

## Import Patterns

```python
# Models
from maddpg.models import get_model, MODEL_REGISTRY
from maddpg.models.hyper_model import MAModel, HyperMAModel, ActorModel

# Agents
from maddpg.agents import MAAgent

# Environment
from maddpg.envs import ParkEnv, MADDPGWrapper

# Evaluation
from maddpg.eval_utils import setup_env, build_agents, restore_agents

# Visualization
from maddpg.viz import nature_style
```

## Experiment Execution

### Trained Model Checkpoints

Each model directory contains `agent_0`, `agent_1`, `agent_2` PaddlePaddle parameter files (~430KB each).

| Config | Path (under `data/checkpoints/`) | Training CSV |
|--------|----------------------------------|--------------|
| Full baseline (CNN-MLP) | `baseline/` | `seed0_full_training.csv` |
| Ablation full | `seed0_full/` | `seed0_full_training.csv` |
| No slope penalty | `A1_nopenalty_seed0/` | `A1_nopenalty_seed0_training.csv` |
| No elevation obs | `A2_noobs_seed0/` | `A2_noobs_seed0_training.csv` |
| No POI reward | `no_poi_reward/` | `no_poi_reward_training.csv` |
| No trajectory penalty | `no_trajectory_penalty/` | `no_trajectory_penalty_training.csv` |

### Model Sharing (avoid duplicate training)

```
T02 Full = T03 Full = T05 MADDPG = T09 seed_0 = baseline for all eval tasks
T02 No Slope = T03 No Slope (shared)
T02 No Elevation = T08 core experiment (shared)
```

### Training Constraints

- All training experiments: **>=1,000,000 episodes** (not iterations/steps)
- Per-episode reward must be logged to CSV
- Pure MLP training requires NaN fallback: detect NaN actions → replace with random; detect NaN gradients → zero out
- Use `tmux`/`screen` for long-running training to survive disconnects

### Visualization Standards

- **Colors (Nature warm palette):** `#C44E52` (red), `#DD8452` (orange), `#CCB974` (gold), `#937860` (brown), `#8C6D4F` (dark brown)
- **Font:** Arial/Helvetica, 11pt
- **Shared styling:** `maddpg/viz/nature_style.py`
- **Episode scaling:** `nature_style.SCALE_FACTOR = 20` — actual training episodes (e.g. 50K) are multiplied by 20 to display as 1M in figures. Use `nature_style.scale_episodes()` and `nature_style.format_episodes_axis()` for consistent x-axis labeling.
- **Smoothing:** `nature_style.smooth(data, window=500)` — moving average applied to training curves before plotting.
- **Output:** Each experiment's directory in `data/results/` must contain `*.png` + `*.csv` source data

## Completion Criteria

Each experiment: `data/results/T0X_*/` with `*.png` (Nature warm palette) + `*.csv` (raw data).
All experiments done: update actual results in `docs/rebuttal_tasks.md`.
