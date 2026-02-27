"""
泛化测试。

加载 Full 模型，5 种环境配置 × 50 episodes。

运行:
    python -m maddpg.experiments.generalization \
        --model_dir data/checkpoints/baseline \
        --output_dir data/results/generalization \
        --episodes 50
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
import pandas as pd

MAX_STEPS = 120


def apply_after_reset(env, config_name):
    base = env.env
    if config_name == 'Reduced Step':
        for i in range(base.num_agents):
            pos = base.agent_positions[i]
            base.agent_positions[i] = pos * 0.7 + base.start_pos * 0.3
    elif config_name == 'More POIs':
        current_pois = len(getattr(base, 'interest_points_centers', [])) if base.num_interest_points is None else base.num_interest_points
        base.num_interest_points = min(current_pois * 2, 30)
    elif config_name == 'Steep Terrain (2x)':
        elev = base.map[1].astype(float)
        mean_e = elev.mean()
        base.map[1] = np.clip((elev - mean_e) * 2 + mean_e, 0, 255).astype(np.uint8)
    elif config_name == 'Flat Terrain':
        base.map[1] = np.full_like(base.map[1], 128)


CONFIG_NAMES = ['Standard', 'Reduced Step', 'More POIs', 'Steep Terrain (2x)', 'Flat Terrain']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    all_results = {}
    baseline_reward = None

    for config_name in CONFIG_NAMES:
        print(f"Evaluating: {config_name} ...")
        env = setup_env()
        agents = build_agents(env)
        restore_agents(agents, args.model_dir)

        results = []
        for ep in range(args.episodes):
            obs_n = env.reset()
            apply_after_reset(env, config_name)
            base_env = env.env
            done = False
            steps = 0
            ep_return = 0.0
            path_len = [0.0] * base_env.num_agents
            prev_pos = [base_env.agent_positions[i].copy() for i in range(base_env.num_agents)]
            while (not done) and steps < MAX_STEPS:
                steps += 1
                action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
                obs_n, reward_n, done_n, _ = env.step(action_n)
                done = all(done_n)
                ep_return += float(np.sum(reward_n))
                for i in range(base_env.num_agents):
                    path_len[i] += float(np.linalg.norm(base_env.agent_positions[i] - prev_pos[i]))
                    prev_pos[i] = base_env.agent_positions[i].copy()
            results.append({
                'config': config_name, 'episode': ep,
                'episode_return': ep_return, 'steps': steps,
                'success': 1 if done else 0, 'path_len_mean': float(np.mean(path_len)),
            })
        all_results[config_name] = results
        mean_r = np.mean([r['episode_return'] for r in results])
        print(f"  Mean reward: {mean_r:.1f}")
        if config_name == 'Standard':
            baseline_reward = mean_r

    # CSV
    csv_path = os.path.join(args.output_dir, 'generalization_data.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'episode', 'reward', 'steps', 'success', 'path_len'])
        for config_name, results in all_results.items():
            for r in results:
                writer.writerow([config_name, r.get('episode', 0), r['episode_return'],
                                 r['steps'], r['success'], r['path_len_mean']])
    print(f"Saved {csv_path}")

    # 性能保持图
    config_names = CONFIG_NAMES
    means = [np.mean([r['episode_return'] for r in all_results[c]]) for c in config_names]
    stds = [np.std([r['episode_return'] for r in all_results[c]]) for c in config_names]
    retention = [(m / baseline_reward * 100) if baseline_reward and abs(baseline_reward) > 1e-6 else 100.0
                 for m in means]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(config_names))

    axes[0].bar(x, means, yerr=stds, capsize=4,
                color=nature_style.COLOR_LIST[:len(config_names)], edgecolor='white', width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(config_names, rotation=20, ha='right', fontsize=9)
    axes[0].set_ylabel('Mean Episode Reward')
    axes[0].set_title('Generalization: Absolute Performance')

    axes[1].bar(x, retention, color=nature_style.COLOR_LIST[:len(config_names)],
                edgecolor='white', width=0.6)
    axes[1].axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(config_names, rotation=20, ha='right', fontsize=9)
    axes[1].set_ylabel('Performance Retention (%)')
    axes[1].set_title('Generalization: Retention vs Standard')

    fig.suptitle('Generalization Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'generalization_analysis.png'))
    plt.close(fig)
    print("Saved generalization_analysis.png")

    summary = []
    for i, c in enumerate(config_names):
        summary.append({'config': c, 'mean_reward': f"{means[i]:.2f}",
                        'std_reward': f"{stds[i]:.2f}", 'retention_pct': f"{retention[i]:.1f}"})
    pd.DataFrame(summary).to_csv(os.path.join(args.output_dir, 'generalization_summary.csv'), index=False)
    print("Saved generalization_summary.csv")


if __name__ == '__main__':
    main()
