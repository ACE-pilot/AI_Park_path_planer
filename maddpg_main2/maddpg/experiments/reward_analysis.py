"""
奖励分量消融 — 训练曲线 + 评估柱状图。

运行:
    python -m maddpg.experiments.reward_analysis \
        --csv_dir data/training_csvs \
        --output_dir data/results/reward_analysis
"""
import os
import argparse
import numpy as np
import pandas as pd

from maddpg.viz import nature_style
nature_style.apply()
import matplotlib.pyplot as plt

CONFIGS = {
    'Full': 'seed0_full_training.csv',
    'No POI Reward': 'no_poi_reward_training.csv',
    'No Traj. Penalty': 'no_trajectory_penalty_training.csv',
    'No Slope Penalty': 'A1_nopenalty_seed0_training.csv',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default='.')
    parser.add_argument('--output_dir', type=str,
                        default='data/results/reward_analysis')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 训练曲线 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (label, csv_name) in enumerate(CONFIGS.items()):
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
    ax.set_title('Reward Component Ablation')
    nature_style.format_episodes_axis(ax)
    ax.legend(loc='lower right')
    fig.savefig(os.path.join(args.output_dir, 'reward_ablation_training.png'))
    plt.close(fig)
    print("Saved reward_ablation_training.png")

    # --- 评估柱状图 ---
    N_EVAL = 1000
    names, means, stds = [], [], []
    for label, csv_name in CONFIGS.items():
        csv_path = os.path.join(args.csv_dir, csv_name)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        tail = df['reward'].values[-N_EVAL:]
        names.append(label)
        means.append(np.mean(tail))
        stds.append(np.std(tail))

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5,
           color=nature_style.COLOR_LIST[:len(names)], edgecolor='white', width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Reward Ablation Evaluation')
    fig.savefig(os.path.join(args.output_dir, 'reward_ablation_eval_bar.png'))
    plt.close(fig)
    print("Saved reward_ablation_eval_bar.png")

    summary = pd.DataFrame({'config': names, 'mean_reward': means, 'std_reward': stds})
    summary.to_csv(os.path.join(args.output_dir, 'reward_ablation_summary.csv'), index=False)
    print("Saved reward_ablation_summary.csv")


if __name__ == '__main__':
    main()
