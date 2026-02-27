"""
可解释性分析（真实 rollout）。

加载 Full 模型，跑 episodes 收集 5000+ (obs, action) 对，然后:
- RandomForestRegressor → 特征重要性 (180 维)
- KMeans → 5 个动作类别
- DecisionTreeClassifier(max_depth=3) → 可解释规则

运行:
    python -m maddpg.experiments.explainability \
        --model_dir data/checkpoints/baseline \
        --output_dir data/results/explainability \
        --min_samples 5000
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

FEATURE_GROUPS = {
    'Self Position': (0, 2),
    'Other Agents': (2, 6),
    'Start Position': (6, 8),
    'End Position': (8, 10),
    'POI Positions': (10, 30),
    'POI Visit Count': (30, 31),
    'Rel. Other Agents': (31, 35),
    'Rel. Start': (35, 37),
    'Rel. End': (37, 39),
    'Rel. POIs': (39, 59),
    'Elevation Map': (59, 180),
}


def collect_rollout_data(env, agents, min_samples):
    all_obs, all_actions = [], []
    episodes = 0
    while len(all_obs) < min_samples:
        obs_n = env.reset()
        done = False
        steps = 0
        while (not done) and steps < MAX_STEPS:
            steps += 1
            action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
            for obs, act in zip(obs_n, action_n):
                all_obs.append(obs.copy())
                all_actions.append(act.copy())
            obs_n, _, done_n, _ = env.step(action_n)
            done = all(done_n)
        episodes += 1
        if episodes % 20 == 0:
            print(f"  Collected {len(all_obs)} samples from {episodes} episodes")
    print(f"Total: {len(all_obs)} samples from {episodes} episodes")
    obs_arr = np.array(all_obs)
    act_arr = np.array(all_actions)
    # 过滤 NaN 样本
    valid = np.all(np.isfinite(obs_arr), axis=1) & np.all(np.isfinite(act_arr), axis=1)
    if not np.all(valid):
        print(f"  Filtered {np.sum(~valid)} NaN samples")
    return obs_arr[valid], act_arr[valid]


def feature_importance_analysis(obs_data, action_data, output_dir):
    from sklearn.ensemble import RandomForestRegressor

    action_magnitude = np.linalg.norm(action_data, axis=1)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(obs_data, action_magnitude)
    importances = rf.feature_importances_

    # CSV
    csv_path = os.path.join(output_dir, 'feature_importance.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature_index', 'importance', 'group'])
        for idx, imp in enumerate(importances):
            group = 'Unknown'
            for gname, (start, end) in FEATURE_GROUPS.items():
                if start <= idx < end:
                    group = gname
                    break
            writer.writerow([idx, f"{imp:.6f}", group])
    print(f"Saved {csv_path}")

    # 分组重要性
    group_importance = {}
    for gname, (start, end) in FEATURE_GROUPS.items():
        group_importance[gname] = float(np.sum(importances[start:end]))

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    groups = list(group_importance.keys())
    values = list(group_importance.values())
    sorted_idx = np.argsort(values)[::-1]
    groups_sorted = [groups[i] for i in sorted_idx]
    values_sorted = [values[i] for i in sorted_idx]
    colors = [nature_style.COLOR_LIST[i % len(nature_style.COLOR_LIST)] for i in range(len(groups_sorted))]

    axes[0].barh(range(len(groups_sorted)), values_sorted, color=colors)
    axes[0].set_yticks(range(len(groups_sorted)))
    axes[0].set_yticklabels(groups_sorted, fontsize=9)
    axes[0].set_xlabel('Feature Importance')
    axes[0].set_title('Feature Group Importance')
    axes[0].invert_yaxis()

    axes[1].bar(range(len(importances)), importances, color=nature_style.COLOR_LIST[0], alpha=0.7, width=1.0)
    for gname, (start, end) in FEATURE_GROUPS.items():
        axes[1].axvline(x=start, color='gray', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Importance')
    axes[1].set_title('Per-Feature Importance (180 dims)')

    fig.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close(fig)
    print("Saved feature_importance.png")

    return importances, group_importance


def action_clustering(action_data, output_dir):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(action_data)
    centers = kmeans.cluster_centers_

    csv_path = os.path.join(output_dir, 'action_clusters.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cluster', 'count', 'pct', 'center_0', 'center_1', 'center_2', 'center_3'])
        for k in range(5):
            mask = labels == k
            writer.writerow([k, int(np.sum(mask)), f"{np.sum(mask)/len(labels)*100:.1f}",
                             f"{centers[k,0]:.3f}", f"{centers[k,1]:.3f}",
                             f"{centers[k,2]:.3f}", f"{centers[k,3]:.3f}"])
    print(f"Saved {csv_path}")
    return labels, centers


def decision_tree_rules(obs_data, action_labels, output_dir):
    from sklearn.tree import DecisionTreeClassifier, export_text

    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(obs_data, action_labels)

    feature_names = []
    for gname, (start, end) in FEATURE_GROUPS.items():
        for i in range(start, end):
            feature_names.append(f"{gname}[{i - start}]")

    rules = export_text(dt, feature_names=feature_names, max_depth=3)
    rules_path = os.path.join(output_dir, 'decision_tree_rules.txt')
    with open(rules_path, 'w') as f:
        f.write(rules)
    print(f"Saved {rules_path}")

    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt, feature_names=feature_names,
              class_names=[f'Cluster {i}' for i in range(5)],
              filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title('Decision Tree Rules (max_depth=3)')
    fig.savefig(os.path.join(output_dir, 'decision_tree_rules.png'), dpi=150)
    plt.close(fig)
    print("Saved decision_tree_rules.png")

    return dt, rules


def decision_surface_plot(obs_data, action_labels, importances, output_dir):
    top2 = np.argsort(importances)[-2:][::-1]
    feature_names = []
    for gname, (start, end) in FEATURE_GROUPS.items():
        for i in range(start, end):
            feature_names.append(f"{gname}[{i - start}]")

    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(5):
        mask = action_labels == k
        ax.scatter(obs_data[mask, top2[0]], obs_data[mask, top2[1]],
                   c=nature_style.COLOR_LIST[k % len(nature_style.COLOR_LIST)],
                   label=f'Cluster {k}', alpha=0.3, s=10)
    ax.set_xlabel(feature_names[top2[0]])
    ax.set_ylabel(feature_names[top2[1]])
    ax.set_title('Decision Surface (Top-2 Features)')
    ax.legend(markerscale=3)
    fig.savefig(os.path.join(output_dir, 'decision_surface.png'))
    plt.close(fig)
    print("Saved decision_surface.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--min_samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    env = setup_env()
    agents = build_agents(env)
    restore_agents(agents, args.model_dir)

    print("=== Collecting rollout data ===")
    obs_data, action_data = collect_rollout_data(env, agents, args.min_samples)

    print("\n=== Feature Importance ===")
    importances, group_imp = feature_importance_analysis(obs_data, action_data, args.output_dir)

    print("\n=== Action Clustering ===")
    labels, centers = action_clustering(action_data, args.output_dir)

    print("\n=== Decision Tree Rules ===")
    dt, rules = decision_tree_rules(obs_data, labels, args.output_dir)

    print("\n=== Decision Surface ===")
    decision_surface_plot(obs_data, labels, importances, args.output_dir)

    print("\n=== Summary ===")
    print(f"Samples: {len(obs_data)}")
    print("Top-3 feature groups:")
    for gname, imp in sorted(group_imp.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {gname}: {imp:.4f}")
    print(f"Decision tree accuracy: {dt.score(obs_data, labels):.3f}")


if __name__ == '__main__':
    main()
