"""
CNN-MLP / MLP / UNet / Attention 架构对比。

训练曲线对比 + 加载 4 种模型评估。

运行:
    python -m maddpg.experiments.architecture_comparison \
        --csv_dir data/training_csvs \
        --model_dirs hyper=data/checkpoints/baseline mlp=data/checkpoints/mlp_model \
                     unet=data/checkpoints/unet_model attention=data/checkpoints/attention_model \
        --output_dir data/results/architecture \
        --episodes 100
"""
import os
import argparse
import csv
import numpy as np
import pandas as pd

import paddle
paddle.set_device('gpu')

from maddpg.eval_utils import setup_env, restore_agents, run_eval_episode
from maddpg.models import MODEL_REGISTRY, get_model
from maddpg.agents import MAAgent
from parl.algorithms import MADDPG

from maddpg.viz import nature_style
nature_style.apply()
import matplotlib.pyplot as plt

MAX_STEPS = 120
CRITIC_LR = 0.001
ACTOR_LR = 0.0001
GAMMA = 0.95
TAU = 0.001
BATCH_SIZE = 1024

ARCH_NAMES = ['hyper', 'mlp', 'unet', 'attention']
ARCH_LABELS = {
    'hyper': 'CNN-MLP (Proposed)',
    'mlp': 'Pure MLP',
    'unet': 'UNet-MLP',
    'attention': 'Attention-MLP',
}

CSV_MAP = {
    'hyper': 'seed0_full_training.csv',
    'mlp': 'mlp_training.csv',
    'unet': 'unet_training.csv',
    'attention': 'attention_training.csv',
}


def build_agents_with_model(env, model_type):
    ModelClass = get_model(model_type)
    act_space_n = [env.action_space[f'agent_{i}'] for i in range(env.n)]
    agents = []
    for i in range(env.n):
        model = ModelClass(
            obs_dim=env.obs_shape_n[i], act_dim=env.act_shape_n[i],
            obs_shape_n=env.obs_shape_n, act_shape_n=env.act_shape_n,
            continuous_actions=True,
        )
        algorithm = MADDPG(
            model, agent_index=i, act_space=act_space_n,
            gamma=GAMMA, tau=TAU, critic_lr=CRITIC_LR, actor_lr=ACTOR_LR,
        )
        agent = MAAgent(
            algorithm, agent_index=i,
            obs_dim_n=env.obs_shape_n, act_dim_n=env.act_shape_n,
            batch_size=BATCH_SIZE,
        )
        agents.append(agent)
    return agents


def parse_model_dirs(args_list):
    dirs = {}
    if args_list is None:
        return dirs
    for item in args_list:
        if '=' in item:
            k, v = item.split('=', 1)
            dirs[k] = v
    return dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default='data/training_csvs')
    parser.add_argument('--model_dirs', nargs='*', default=[])
    parser.add_argument('--output_dir', type=str, default='data/results/architecture')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    model_dirs = parse_model_dirs(args.model_dirs)

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, arch in enumerate(ARCH_NAMES):
        csv_path = os.path.join(args.csv_dir, CSV_MAP.get(arch, ''))
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping training curve for {arch}")
            continue
        df = pd.read_csv(csv_path)
        episodes = nature_style.scale_episodes(df['episode'].values)
        reward = nature_style.smooth(df['reward'].values)
        episodes = episodes[:len(reward)]
        ax.plot(episodes, reward, color=nature_style.COLOR_LIST[idx],
                label=ARCH_LABELS[arch], linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Architecture Training Curves')
    nature_style.format_episodes_axis(ax)
    ax.legend(loc='lower right')
    fig.savefig(os.path.join(args.output_dir, 'architecture_training_curve.png'))
    plt.close(fig)
    print("Saved architecture_training_curve.png")

    all_rows = []
    arch_means, arch_stds = [], []
    eval_labels = []

    for arch in ARCH_NAMES:
        mdir = model_dirs.get(arch)
        if not mdir or not os.path.exists(os.path.join(mdir, 'agent_0')):
            print(f"Warning: model for {arch} not found at {mdir}, skipping eval")
            continue

        print(f"Evaluating {ARCH_LABELS[arch]} ...")
        env = setup_env()
        agents = build_agents_with_model(env, arch)
        restore_agents(agents, mdir)

        rewards = []
        for ep in range(args.episodes):
            res = run_eval_episode(env, agents, MAX_STEPS)
            res['architecture'] = arch
            res['episode'] = ep
            all_rows.append(res)
            rewards.append(res['episode_return'])

        eval_labels.append(ARCH_LABELS[arch])
        arch_means.append(np.mean(rewards))
        arch_stds.append(np.std(rewards))
        print(f"  Mean: {arch_means[-1]:.2f} +/- {arch_stds[-1]:.2f}")

    if all_rows:
        csv_path = os.path.join(args.output_dir, 'architecture_comparison_data.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['architecture', 'episode', 'reward', 'steps', 'success', 'path_len'])
            for r in all_rows:
                writer.writerow([r['architecture'], r['episode'], r['episode_return'],
                                 r['steps'], r['success'], r['path_len_mean']])
        print(f"Saved {csv_path}")

    if eval_labels:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(eval_labels))
        ax.bar(x, arch_means, yerr=arch_stds, capsize=5,
               color=nature_style.COLOR_LIST[:len(eval_labels)], edgecolor='white', width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(eval_labels, rotation=15, ha='right')
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title('Architecture Evaluation')
        fig.savefig(os.path.join(args.output_dir, 'architecture_eval_bar.png'))
        plt.close(fig)
        print("Saved architecture_eval_bar.png")


if __name__ == '__main__':
    main()
