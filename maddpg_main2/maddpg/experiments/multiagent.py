"""
MADDPG vs Independent DDPG vs Random 对比评估。

运行:
    python -m maddpg.experiments.multiagent \
        --maddpg_dir data/checkpoints/baseline \
        --indddpg_dir data/checkpoints/independent_ddpg \
        --output_dir data/results/multiagent \
        --episodes 100
"""
import os
import argparse
import csv
import numpy as np

import paddle
paddle.set_device('gpu')

from maddpg.eval_utils import setup_env, build_agents, restore_agents, run_eval_episode
from maddpg.scripts.train_independent_ddpg import IndependentMAModel
from parl.algorithms import MADDPG
from maddpg.agents import MAAgent

from maddpg.viz import nature_style
nature_style.apply()
import matplotlib.pyplot as plt
import pandas as pd

MAX_STEPS = 120
CRITIC_LR = 0.001
ACTOR_LR = 0.0001
GAMMA = 0.95
TAU = 0.001
BATCH_SIZE = 1024


def build_independent_agents(env):
    act_space_n = [env.action_space[f'agent_{i}'] for i in range(env.n)]
    agents = []
    for i in range(env.n):
        model = IndependentMAModel(
            obs_dim=env.obs_shape_n[i], act_dim=env.act_shape_n[i],
            obs_shape_n=env.obs_shape_n, act_shape_n=env.act_shape_n,
            continuous_actions=True
        )
        algorithm = MADDPG(
            model, agent_index=i, act_space=act_space_n,
            gamma=GAMMA, tau=TAU, critic_lr=CRITIC_LR, actor_lr=ACTOR_LR
        )
        agent = MAAgent(
            algorithm, agent_index=i,
            obs_dim_n=env.obs_shape_n, act_dim_n=env.act_shape_n,
            batch_size=BATCH_SIZE
        )
        agents.append(agent)
    return agents


def eval_random(env, episodes, max_steps):
    results = []
    for _ in range(episodes):
        env.reset()
        done = False
        steps = 0
        ep_return = 0.0
        while (not done) and steps < max_steps:
            steps += 1
            action_n = [np.random.uniform(0, 1, size=4) for _ in range(env.n)]
            _, reward_n, done_n, _ = env.step(action_n)
            done = all(done_n)
            ep_return += float(np.sum(reward_n))
        results.append({'episode_return': ep_return, 'steps': steps, 'success': 1 if done else 0})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maddpg_dir', type=str, default='./model')
    parser.add_argument('--indddpg_dir', type=str, default='./model_independent_ddpg')
    parser.add_argument('--output_dir', type=str,
                        default='data/results/multiagent')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    env = setup_env()

    # MADDPG
    print("Evaluating MADDPG...")
    maddpg_agents = build_agents(env)
    restore_agents(maddpg_agents, args.maddpg_dir)
    maddpg_results = [run_eval_episode(env, maddpg_agents, MAX_STEPS) for _ in range(args.episodes)]
    print(f"  Mean: {np.mean([r['episode_return'] for r in maddpg_results]):.1f}")

    # Independent DDPG
    print("Evaluating Independent DDPG...")
    if os.path.exists(os.path.join(args.indddpg_dir, 'agent_0')):
        ind_agents = build_independent_agents(env)
        restore_agents(ind_agents, args.indddpg_dir)
        ind_results = [run_eval_episode(env, ind_agents, MAX_STEPS) for _ in range(args.episodes)]
        print(f"  Mean: {np.mean([r['episode_return'] for r in ind_results]):.1f}")
    else:
        print(f"  Warning: model not found at {args.indddpg_dir}")
        ind_results = [{'episode_return': 0, 'steps': MAX_STEPS, 'success': 0}] * args.episodes

    # Random
    print("Evaluating Random...")
    random_results = eval_random(env, args.episodes, MAX_STEPS)
    print(f"  Mean: {np.mean([r['episode_return'] for r in random_results]):.1f}")

    # 保存 CSV
    csv_path = os.path.join(args.output_dir, 'comparison_data.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'episode', 'reward', 'steps', 'success'])
        for ep, r in enumerate(maddpg_results):
            writer.writerow(['MADDPG', ep, r['episode_return'], r['steps'], r['success']])
        for ep, r in enumerate(ind_results):
            writer.writerow(['IndDDPG', ep, r['episode_return'], r['steps'], r['success']])
        for ep, r in enumerate(random_results):
            writer.writerow(['Random', ep, r['episode_return'], r['steps'], r['success']])
    print(f"Saved {csv_path}")

    # 柱状图
    methods = ['MADDPG', 'IndDDPG', 'Random']
    all_res = [maddpg_results, ind_results, random_results]
    means = [np.mean([r['episode_return'] for r in res]) for res in all_res]
    stds = [np.std([r['episode_return'] for r in res]) for res in all_res]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=5,
           color=nature_style.COLOR_LIST[:3], edgecolor='white', width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Multi-Agent Method Comparison')
    fig.savefig(os.path.join(args.output_dir, 'method_comparison.png'))
    plt.close(fig)
    print("Saved method_comparison.png")

    # 训练曲线对比
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    csv_dir = os.path.join(project_root, 'data', 'training_csvs')
    maddpg_csv = os.path.join(csv_dir, 'seed0_full_training.csv')
    indddpg_csv = os.path.join(csv_dir, 'independent_ddpg_training.csv')

    fig, ax = plt.subplots(figsize=(8, 5))
    if os.path.exists(maddpg_csv):
        df = pd.read_csv(maddpg_csv)
        eps = nature_style.scale_episodes(df['episode'].values)
        rew = nature_style.smooth(df['reward'].values)
        ax.plot(eps[:len(rew)], rew, color=nature_style.COLOR_LIST[0], label='MADDPG', linewidth=1.5)
    if os.path.exists(indddpg_csv):
        df = pd.read_csv(indddpg_csv)
        eps = nature_style.scale_episodes(df['episode'].values)
        rew = nature_style.smooth(df['reward'].values)
        ax.plot(eps[:len(rew)], rew, color=nature_style.COLOR_LIST[1], label='Independent DDPG', linewidth=1.5)

    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Training Curve Comparison')
    nature_style.format_episodes_axis(ax)
    ax.legend()
    fig.savefig(os.path.join(args.output_dir, 'training_comparison.png'))
    plt.close(fig)
    print("Saved training_comparison.png")


if __name__ == '__main__':
    main()
