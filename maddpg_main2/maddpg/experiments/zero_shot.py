"""
零样本迁移。

加载 Full 模型，8 个随机种子地图 × 30 episodes。

运行:
    python -m maddpg.experiments.zero_shot \
        --model_dir data/checkpoints/baseline \
        --output_dir data/results/zero_shot \
        --episodes_per_map 30
"""
import os
import argparse
import csv
import numpy as np

import paddle
paddle.set_device('gpu')

from maddpg.eval_utils import setup_env, build_agents, restore_agents, run_eval_episode

from maddpg.viz import nature_style
nature_style.apply()
import matplotlib.pyplot as plt

MAX_STEPS = 120
MAP_SEEDS = [200, 300, 400, 500, 600, 700, 800, 900]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--episodes_per_map', type=int, default=30)
    parser.add_argument('--map_seeds', type=int, nargs='+', default=MAP_SEEDS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []

    for map_seed in args.map_seeds:
        print(f"Map seed {map_seed}:")
        np.random.seed(map_seed)

        env = setup_env()
        agents = build_agents(env)
        restore_agents(agents, args.model_dir)

        rewards = []
        for ep in range(args.episodes_per_map):
            np.random.seed(map_seed * 10000 + ep)
            res = run_eval_episode(env, agents, MAX_STEPS)
            res['map_seed'] = map_seed
            res['episode'] = ep
            all_rows.append(res)
            rewards.append(res['episode_return'])

        print(f"  Mean: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")

    # CSV
    csv_path = os.path.join(args.output_dir, 'zero_shot_data.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['map_seed', 'episode', 'reward', 'steps', 'success', 'path_len'])
        for r in all_rows:
            writer.writerow([r['map_seed'], r['episode'], r['episode_return'],
                             r['steps'], r['success'], r['path_len_mean']])
    print(f"Saved {csv_path}")

    # 跨地图性能图
    seed_labels = [str(s) for s in args.map_seeds]
    seed_means, seed_stds = [], []
    for seed in args.map_seeds:
        rows = [r for r in all_rows if r['map_seed'] == seed]
        seed_means.append(np.mean([r['episode_return'] for r in rows]))
        seed_stds.append(np.std([r['episode_return'] for r in rows]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(seed_labels))
    colors = [nature_style.COLOR_LIST[i % len(nature_style.COLOR_LIST)] for i in range(len(seed_labels))]

    axes[0].bar(x, seed_means, yerr=seed_stds, capsize=4, color=colors, edgecolor='white', width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Seed {s}' for s in seed_labels], rotation=30, ha='right')
    axes[0].set_ylabel('Mean Episode Reward')
    axes[0].set_title('Zero-Shot: Per-Map Performance')

    box_data = [[r['episode_return'] for r in all_rows if r['map_seed'] == seed] for seed in args.map_seeds]
    bp = axes[1].boxplot(box_data, patch_artist=True, labels=[f'S{s}' for s in seed_labels])
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Episode Reward')
    axes[1].set_title('Zero-Shot: Reward Distribution')

    fig.suptitle('Zero-Shot Transfer Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'zero_shot_analysis.png'))
    plt.close(fig)
    print("Saved zero_shot_analysis.png")

    overall_mean = np.mean([r['episode_return'] for r in all_rows])
    overall_std = np.std([r['episode_return'] for r in all_rows])
    print(f"\n=== Summary ===")
    print(f"Overall: {overall_mean:.1f} +/- {overall_std:.1f}")
    print(f"Success rate: {np.mean([r['success'] for r in all_rows]) * 100:.1f}%")
    print(f"Cross-map std: {np.std(seed_means):.1f}")


if __name__ == '__main__':
    main()
