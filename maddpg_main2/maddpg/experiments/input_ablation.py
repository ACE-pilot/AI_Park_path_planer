"""
输入通道消融 — 敏感性分析。

加载 Full 模型，5 种遮蔽配置 × 100 episodes。

运行:
    python -m maddpg.experiments.input_ablation \
        --model_dir data/checkpoints/baseline \
        --output_dir data/results/input_ablation \
        --episodes 100
"""
import os
import argparse
import csv
import numpy as np

import paddle
paddle.set_device('gpu')

from maddpg.eval_utils import setup_env, build_agents, restore_agents

from maddpg.viz import nature_style
nature_style.apply()
import matplotlib.pyplot as plt

MAX_STEPS = 120

# 观测结构 (180 维):
# [0:2]   自身位置
# [2:6]   其他 agent 位置
# [6:8]   起点
# [8:10]  终点
# [10:30] POI 位置
# [30:31] POI 访问次数
# [31:35] 相对其他 agent
# [35:37] 相对起点
# [37:39] 相对终点
# [39:59] 相对 POI
# [59:180] 局部高程图 (11×11)

MASKING_CONFIGS = {
    'Full (baseline)': None,
    'No Elevation': (59, 180),
    'No POI Info': (10, 31),
    'No Other Agents': (2, 6),
    'No Relative Info': (31, 59),
}


def run_masked_episode(env, agents, max_steps, mask_range=None):
    obs_n = env.reset()
    done = False
    steps = 0
    ep_return = 0.0
    while (not done) and steps < max_steps:
        steps += 1
        masked_obs_n = []
        for obs in obs_n:
            obs_copy = obs.copy()
            if mask_range is not None:
                obs_copy[mask_range[0]:mask_range[1]] = 0.0
            masked_obs_n.append(obs_copy)
        action_n = [agent.predict(obs) for agent, obs in zip(agents, masked_obs_n)]
        obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)
        ep_return += float(np.sum(reward_n))
    return {'episode_return': ep_return, 'steps': steps, 'success': 1 if done else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    env = setup_env()
    agents = build_agents(env)
    restore_agents(agents, args.model_dir)

    all_results = {}
    for config_name, mask_range in MASKING_CONFIGS.items():
        print(f"Evaluating: {config_name} ...")
        results = []
        for ep in range(args.episodes):
            res = run_masked_episode(env, agents, MAX_STEPS, mask_range)
            res['config'] = config_name
            res['episode'] = ep
            results.append(res)
        all_results[config_name] = results
        print(f"  Mean reward: {np.mean([r['episode_return'] for r in results]):.1f}")

    # CSV
    csv_path = os.path.join(args.output_dir, 'input_ablation_data.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'episode', 'reward', 'steps', 'success'])
        for config_name, results in all_results.items():
            for r in results:
                writer.writerow([config_name, r['episode'], r['episode_return'], r['steps'], r['success']])
    print(f"Saved {csv_path}")

    # 柱状图
    baseline_mean = np.mean([r['episode_return'] for r in all_results['Full (baseline)']])
    config_names = list(MASKING_CONFIGS.keys())
    means = [np.mean([r['episode_return'] for r in all_results[c]]) for c in config_names]
    stds = [np.std([r['episode_return'] for r in all_results[c]]) for c in config_names]
    drops = [(baseline_mean - m) / abs(baseline_mean) * 100 if baseline_mean != 0 else 0 for m in means]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(config_names))

    axes[0].bar(x, means, yerr=stds, capsize=4,
                color=nature_style.COLOR_LIST[:len(config_names)], edgecolor='white', width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(config_names, rotation=20, ha='right', fontsize=9)
    axes[0].set_ylabel('Mean Episode Reward')
    axes[0].set_title('Input Ablation: Absolute Performance')

    axes[1].bar(x[1:], drops[1:], color=nature_style.COLOR_LIST[1:len(config_names)],
                edgecolor='white', width=0.6)
    axes[1].set_xticks(x[1:])
    axes[1].set_xticklabels(config_names[1:], rotation=20, ha='right', fontsize=9)
    axes[1].set_ylabel('Performance Drop (%)')
    axes[1].set_title('Input Channel Sensitivity')

    fig.suptitle('Input Channel Ablation', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'input_ablation_comparison.png'))
    plt.close(fig)
    print("Saved input_ablation_comparison.png")

    # 热力图
    metrics_labels = ['Reward', 'Success %', 'Avg Steps']
    data_matrix = []
    for c in config_names:
        results = all_results[c]
        data_matrix.append([
            np.mean([r['episode_return'] for r in results]),
            np.mean([r['success'] for r in results]) * 100,
            np.mean([r['steps'] for r in results]),
        ])
    data_matrix = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data_matrix, cmap='YlOrRd_r', aspect='auto')
    ax.set_xticks(range(len(metrics_labels)))
    ax.set_xticklabels(metrics_labels)
    ax.set_yticks(range(len(config_names)))
    ax.set_yticklabels(config_names)
    for i in range(len(config_names)):
        for j in range(len(metrics_labels)):
            ax.text(j, i, f'{data_matrix[i, j]:.1f}', ha='center', va='center', fontsize=9)
    fig.colorbar(im)
    ax.set_title('Input Ablation Heatmap')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'input_ablation_heatmap.png'))
    plt.close(fig)
    print("Saved input_ablation_heatmap.png")


if __name__ == '__main__':
    main()
