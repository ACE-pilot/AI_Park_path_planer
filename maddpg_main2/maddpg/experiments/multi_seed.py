"""
多种子统计验证。

读取多 seed 训练 CSV + 加载多 seed 模型评估 → mean±std。

运行:
    python -m maddpg.experiments.multi_seed \
        --csv_dir data/training_csvs \
        --model_dirs data/checkpoints/seed_0 data/checkpoints/seed_1 data/checkpoints/seed_2 \
        --output_dir data/results/multi_seed \
        --episodes 200
"""
import os
import argparse
import csv
import numpy as np
import pandas as pd

import paddle
paddle.set_device('gpu')

from maddpg.eval_utils import setup_env, build_agents, restore_agents, run_eval_episode

from maddpg.viz import nature_style
nature_style.apply()
import matplotlib.pyplot as plt

MAX_STEPS = 120

SEED_CSV_MAP = {
    'seed_0': 'seed0_full_training.csv',
    'seed_1': 'seed1_training.csv',
    'seed_2': 'seed2_training.csv',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default='data/training_csvs')
    parser.add_argument('--model_dirs', nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, default='data/results/multi_seed')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.random_seed)

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (label, csv_name) in enumerate(SEED_CSV_MAP.items()):
        csv_path = os.path.join(args.csv_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        episodes = nature_style.scale_episodes(df['episode'].values)
        reward = nature_style.smooth(df['reward'].values)
        episodes = episodes[:len(reward)]
        ax.plot(episodes, reward, color=nature_style.COLOR_LIST[idx],
                label=label, linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Multi-Seed Training Curves')
    nature_style.format_episodes_axis(ax)
    ax.legend(loc='lower right')
    fig.savefig(os.path.join(args.output_dir, 'multi_seed_training.png'))
    plt.close(fig)
    print("Saved multi_seed_training.png")

    all_rows = []
    seed_means = []

    for seed_idx, mdir in enumerate(args.model_dirs):
        seed_label = f'seed_{seed_idx}'
        agent_file = os.path.join(mdir, 'agent_0')
        if not os.path.exists(agent_file):
            print(f"Warning: {agent_file} not found, skipping")
            continue

        print(f"Evaluating {seed_label} from {mdir} ...")
        env = setup_env()
        agents = build_agents(env)
        restore_agents(agents, mdir)

        rewards = []
        for ep in range(args.episodes):
            res = run_eval_episode(env, agents, MAX_STEPS)
            res['seed'] = seed_label
            res['episode'] = ep
            all_rows.append(res)
            rewards.append(res['episode_return'])

        mean_r = np.mean(rewards)
        seed_means.append(mean_r)
        print(f"  Mean: {mean_r:.2f} +/- {np.std(rewards):.2f}")

    if not all_rows:
        print("No models evaluated.")
        return

    csv_path = os.path.join(args.output_dir, 'multi_seed_data.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'episode', 'reward', 'steps', 'success', 'path_len'])
        for r in all_rows:
            writer.writerow([r['seed'], r['episode'], r['episode_return'],
                             r['steps'], r['success'], r['path_len_mean']])
    print(f"Saved {csv_path}")

    seeds_in_data = sorted(set(r['seed'] for r in all_rows))
    box_data = [[r['episode_return'] for r in all_rows if r['seed'] == s] for s in seeds_in_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bp = axes[0].boxplot(box_data, patch_artist=True, labels=seeds_in_data)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(nature_style.COLOR_LIST[i % len(nature_style.COLOR_LIST)])
        patch.set_alpha(0.7)
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('Multi-Seed Evaluation Distribution')

    seed_labels = seeds_in_data
    means = [np.mean(d) for d in box_data]
    stds = [np.std(d) for d in box_data]
    x = np.arange(len(seed_labels))
    axes[1].bar(x, means, yerr=stds, capsize=5,
                color=nature_style.COLOR_LIST[:len(seed_labels)], edgecolor='white', width=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(seed_labels)
    axes[1].set_ylabel('Mean Episode Reward')
    axes[1].set_title('Multi-Seed Summary')

    fig.suptitle('Statistical Validation', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'multi_seed_evaluation.png'))
    plt.close(fig)
    print("Saved multi_seed_evaluation.png")

    overall_mean = np.mean(seed_means)
    overall_std = np.std(seed_means)
    print(f"\n=== Cross-Seed Summary ===")
    print(f"Mean: {overall_mean:.2f} +/- {overall_std:.2f}")
    print(f"CV: {overall_std / abs(overall_mean) * 100:.1f}%")

    summary = pd.DataFrame({
        'seed': seed_labels,
        'mean_reward': [f"{m:.2f}" for m in means],
        'std_reward': [f"{s:.2f}" for s in stds],
    })
    summary.to_csv(os.path.join(args.output_dir, 'multi_seed_summary.csv'), index=False)
    print("Saved multi_seed_summary.csv")


if __name__ == '__main__':
    main()
