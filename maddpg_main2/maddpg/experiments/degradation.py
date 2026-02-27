"""
退化行为检测。

加载 Full 模型，跑 200 episodes，检测:
- 微步: 位移 < 0.5
- 振荡: 连续移动夹角 > 120°
- 卡住: 3步累计位移 < 1.0
排除到达终点后的步。

运行:
    python -m maddpg.experiments.degradation \
        --model_dir data/checkpoints/baseline \
        --output_dir data/results/degradation \
        --episodes 200
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
MICRO_THRESHOLD = 0.5
OSCILLATION_ANGLE = 120.0
STUCK_WINDOW = 3
STUCK_THRESHOLD = 1.0


def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def detect_degradation(positions_history, end_positions, arrival_radius=3.0):
    results = []
    for agent_idx, traj in enumerate(positions_history):
        end_pos = end_positions[agent_idx]
        arrived_step = None
        for t, pos in enumerate(traj):
            if np.linalg.norm(pos - end_pos) < arrival_radius:
                arrived_step = t
                break

        active_end = arrived_step if arrived_step is not None else len(traj)
        total_active = max(active_end - 1, 1)

        micro_count = 0
        osc_count = 0
        stuck_count = 0

        for t in range(1, active_end):
            disp = np.linalg.norm(traj[t] - traj[t - 1])
            if disp < MICRO_THRESHOLD:
                micro_count += 1

        for t in range(2, active_end):
            v1 = traj[t - 1] - traj[t - 2]
            v2 = traj[t] - traj[t - 1]
            if angle_between(v1, v2) > OSCILLATION_ANGLE:
                osc_count += 1

        for t in range(STUCK_WINDOW, active_end):
            cum_disp = sum(
                np.linalg.norm(traj[t - j] - traj[t - j - 1])
                for j in range(STUCK_WINDOW)
            )
            if cum_disp < STUCK_THRESHOLD:
                stuck_count += 1

        results.append({
            'agent': agent_idx,
            'active_steps': total_active,
            'micro_steps': micro_count,
            'oscillations': osc_count,
            'stuck_events': stuck_count,
            'micro_pct': micro_count / total_active * 100,
            'osc_pct': osc_count / max(total_active - 1, 1) * 100,
            'stuck_pct': stuck_count / max(total_active - STUCK_WINDOW + 1, 1) * 100,
            'arrived': arrived_step is not None,
            'arrival_step': arrived_step if arrived_step is not None else -1,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    env = setup_env()
    agents = build_agents(env)
    restore_agents(agents, args.model_dir)

    base_env = env.env
    all_rows = []

    for ep in range(args.episodes):
        obs_n = env.reset()
        done = False
        steps = 0

        positions_history = [[] for _ in range(base_env.num_agents)]
        for i in range(base_env.num_agents):
            positions_history[i].append(base_env.agent_positions[i].copy())

        while (not done) and steps < MAX_STEPS:
            steps += 1
            action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
            obs_n, reward_n, done_n, _ = env.step(action_n)
            done = all(done_n)
            for i in range(base_env.num_agents):
                positions_history[i].append(base_env.agent_positions[i].copy())

        end_positions = [base_env.end_pos.copy() for _ in range(base_env.num_agents)]
        deg_results = detect_degradation(positions_history, end_positions)

        for r in deg_results:
            r['episode'] = ep
            all_rows.append(r)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{args.episodes}")

    # 保存 CSV
    csv_path = os.path.join(args.output_dir, 'degradation_data.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {csv_path}")

    # 饼图
    total_micro = sum(r['micro_steps'] for r in all_rows)
    total_osc = sum(r['oscillations'] for r in all_rows)
    total_stuck = sum(r['stuck_events'] for r in all_rows)
    total_normal = sum(r['active_steps'] for r in all_rows) - total_micro - total_osc - total_stuck
    total_normal = max(total_normal, 0)

    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [total_normal, total_micro, total_osc, total_stuck]
    labels_pie = ['Normal', 'Micro-steps', 'Oscillation', 'Stuck']
    colors = nature_style.COLOR_LIST[:4]
    ax.pie(sizes, labels=labels_pie, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Step Classification')
    fig.savefig(os.path.join(args.output_dir, 'degradation_pie.png'))
    plt.close(fig)
    print("Saved degradation_pie.png")

    # 效率分析图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    arrival_steps = [r['arrival_step'] for r in all_rows if r['arrived']]
    if arrival_steps:
        axes[0].boxplot(arrival_steps, patch_artist=True,
                        boxprops=dict(facecolor=nature_style.COLOR_LIST[0], alpha=0.7))
        axes[0].set_ylabel('Steps to Arrival')
        axes[0].set_title('Arrival Efficiency')

    for agent_idx in range(3):
        agent_rows = [r for r in all_rows if r['agent'] == agent_idx]
        micro_pct = np.mean([r['micro_pct'] for r in agent_rows])
        osc_pct = np.mean([r['osc_pct'] for r in agent_rows])
        stuck_pct = np.mean([r['stuck_pct'] for r in agent_rows])
        width = 0.25
        axes[1].bar(agent_idx - width, micro_pct, width, color=nature_style.COLOR_LIST[1],
                     label='Micro' if agent_idx == 0 else '')
        axes[1].bar(agent_idx, osc_pct, width, color=nature_style.COLOR_LIST[2],
                     label='Oscillation' if agent_idx == 0 else '')
        axes[1].bar(agent_idx + width, stuck_pct, width, color=nature_style.COLOR_LIST[3],
                     label='Stuck' if agent_idx == 0 else '')

    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(['Agent 0', 'Agent 1', 'Agent 2'])
    axes[1].set_ylabel('Degradation Rate (%)')
    axes[1].set_title('Per-Agent Degradation')
    axes[1].legend()

    fig.suptitle('Degradation Behavior Analysis')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'degradation_analysis.png'))
    plt.close(fig)
    print("Saved degradation_analysis.png")

    print(f"\n=== Summary ({args.episodes} episodes) ===")
    print(f"Arrival rate: {sum(1 for r in all_rows if r['arrived'])} / {len(all_rows)}")
    print(f"Mean micro-step rate: {np.mean([r['micro_pct'] for r in all_rows]):.1f}%")
    print(f"Mean oscillation rate: {np.mean([r['osc_pct'] for r in all_rows]):.1f}%")
    print(f"Mean stuck rate: {np.mean([r['stuck_pct'] for r in all_rows]):.1f}%")


if __name__ == '__main__':
    main()
