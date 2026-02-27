"""
协作可视化。

加载 Full 模型，跑 20 episodes，记录位置、距离、POI 访问。

运行:
    python -m maddpg.experiments.cooperation \
        --model_dir data/checkpoints/baseline \
        --output_dir data/results/cooperation \
        --episodes 20
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
AGENT_COLORS = ['#C44E52', '#DD8452', '#CCB974']
AGENT_LABELS = ['Agent 0', 'Agent 1', 'Agent 2']


def run_episode_with_tracking(env, agents, max_steps):
    obs_n = env.reset()
    base_env = env.env
    done = False
    steps = 0
    ep_return = 0.0

    positions = [[] for _ in range(base_env.num_agents)]
    inter_dists = []
    poi_visits = [0] * base_env.num_agents

    for i in range(base_env.num_agents):
        positions[i].append(base_env.agent_positions[i].copy())

    while (not done) and steps < max_steps:
        steps += 1
        action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
        obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)
        ep_return += float(np.sum(reward_n))

        step_dists = []
        for i in range(base_env.num_agents):
            positions[i].append(base_env.agent_positions[i].copy())
            for j in range(i + 1, base_env.num_agents):
                d = np.linalg.norm(base_env.agent_positions[i] - base_env.agent_positions[j])
                step_dists.append(d)
        inter_dists.append(step_dists)

        if hasattr(base_env, 'interest_points_visit_count'):
            poi_visits_total = sum(base_env.interest_points_visit_count.values())
            for i in range(base_env.num_agents):
                poi_visits[i] = poi_visits_total

    return {
        'positions': positions,
        'inter_dists': inter_dists,
        'poi_visits': poi_visits,
        'episode_return': ep_return,
        'steps': steps,
        'success': done,
        'start_positions': [base_env.start_pos.copy() for _ in range(base_env.num_agents)],
        'end_positions': [base_env.end_pos.copy() for _ in range(base_env.num_agents)],
        'poi_positions': [np.array(p) for p in base_env.interest_points_centers] if hasattr(base_env, 'interest_points_centers') else [],
        'map': base_env.map.copy(),
        'map_size': base_env.map_size,
    }


def plot_trajectory(ep_data, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    elev = ep_data['map'][1]
    ax.imshow(elev, cmap='terrain', alpha=0.5, origin='lower')

    for i in range(3):
        traj = np.array(ep_data['positions'][i])
        ax.plot(traj[:, 0], traj[:, 1], color=AGENT_COLORS[i],
                linewidth=2, label=AGENT_LABELS[i], alpha=0.8)
        ax.scatter(traj[0, 0], traj[0, 1], color=AGENT_COLORS[i],
                   marker='o', s=100, zorder=5, edgecolors='black')
        ax.scatter(traj[-1, 0], traj[-1, 1], color=AGENT_COLORS[i],
                   marker='*', s=150, zorder=5, edgecolors='black')

    for i in range(3):
        start = ep_data['start_positions'][i]
        end = ep_data['end_positions'][i]
        ax.scatter(start[0], start[1], color='green', marker='s', s=80, zorder=6)
        ax.scatter(end[0], end[1], color='red', marker='D', s=80, zorder=6)

    pois = ep_data['poi_positions']
    if len(pois) > 0:
        pois = np.array(pois)
        ax.scatter(pois[:, 0], pois[:, 1], color='gold', marker='^',
                   s=60, zorder=4, edgecolors='black', label='POIs')

    ax.set_xlim(0, ep_data['map_size'])
    ax.set_ylim(0, ep_data['map_size'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Agent Trajectories (Best Episode)')
    ax.legend(loc='upper right')
    fig.savefig(output_path)
    plt.close(fig)


def plot_distances(ep_data, output_path):
    dists = np.array(ep_data['inter_dists'])
    pair_labels = ['Agent 0-1', 'Agent 0-2', 'Agent 1-2']
    fig, ax = plt.subplots(figsize=(8, 4))
    for j in range(min(dists.shape[1], 3)):
        ax.plot(range(len(dists)), dists[:, j],
                color=nature_style.COLOR_LIST[j], label=pair_labels[j], linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Inter-Agent Distance')
    ax.set_title('Inter-Agent Distance Over Time')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_radar(all_episodes, output_path):
    metrics = {
        'Success Rate': np.mean([e['success'] for e in all_episodes]),
        'Avg Reward': np.mean([e['episode_return'] for e in all_episodes]),
        'Spatial Coverage': np.mean([
            np.mean([np.std(np.array(e['positions'][i])[:, 0]) + np.std(np.array(e['positions'][i])[:, 1])
                      for i in range(3)])
            for e in all_episodes
        ]),
        'Avg Distance': np.mean([
            np.mean(e['inter_dists']) if len(e['inter_dists']) > 0 else 0
            for e in all_episodes
        ]),
        'POI Coverage': np.mean([np.sum(e['poi_visits']) for e in all_episodes]),
    }

    labels = list(metrics.keys())
    values = list(metrics.values())
    max_vals = [max(abs(v), 1e-6) for v in values]
    norm_values = [v / m for v, m in zip(values, max_vals)]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    norm_values += norm_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, norm_values, color=nature_style.COLOR_LIST[0], alpha=0.25)
    ax.plot(angles, norm_values, color=nature_style.COLOR_LIST[0], linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('Cooperation Metrics', pad=20)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    env = setup_env()
    agents = build_agents(env)
    restore_agents(agents, args.model_dir)

    all_episodes = []
    for ep in range(args.episodes):
        ep_data = run_episode_with_tracking(env, agents, MAX_STEPS)
        all_episodes.append(ep_data)
        print(f"Episode {ep + 1}/{args.episodes}, reward: {ep_data['episode_return']:.1f}")

    best_idx = max(range(len(all_episodes)), key=lambda i: all_episodes[i]['episode_return'])
    best = all_episodes[best_idx]
    print(f"Best episode: {best_idx}, reward: {best['episode_return']:.1f}")

    # CSV
    csv_path = os.path.join(args.output_dir, 'cooperation_data.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'steps', 'success', 'mean_inter_dist', 'poi_visits_total'])
        for ep, data in enumerate(all_episodes):
            mean_dist = np.mean(data['inter_dists']) if data['inter_dists'] else 0
            writer.writerow([ep, data['episode_return'], data['steps'],
                             int(data['success']), f"{mean_dist:.2f}", sum(data['poi_visits'])])
    print(f"Saved {csv_path}")

    plot_trajectory(best, os.path.join(args.output_dir, 'cooperation_trajectory.png'))
    print("Saved cooperation_trajectory.png")
    plot_distances(best, os.path.join(args.output_dir, 'agent_distances.png'))
    print("Saved agent_distances.png")
    plot_radar(all_episodes, os.path.join(args.output_dir, 'cooperation_radar.png'))
    print("Saved cooperation_radar.png")


if __name__ == '__main__':
    main()
