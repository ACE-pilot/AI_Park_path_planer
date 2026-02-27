"""
轨迹后处理实验。

7 种方法: Raw, DP(0.5/1.0/2.0), B-spline, Constrained, MovAvg
包含端到端计时、GeoJSON 导出、坡度剖面对比。

运行:
    python -m maddpg.experiments.trajectory \
        --model_dir data/checkpoints/baseline \
        --output_dir data/results/trajectory \
        --episodes 100
"""
import os
import argparse
import csv
import json
import time
import numpy as np

import paddle
paddle.set_device('gpu')

from maddpg.eval_utils import setup_env, build_agents, restore_agents, get_elevation, compute_slope

from maddpg.viz import nature_style
nature_style.apply()
import matplotlib.pyplot as plt

MAX_STEPS = 120
ADA_THRESHOLD = 8.33  # ADA 坡度阈值 (%)


# ========== 后处理方法 ==========

def method_raw(traj):
    """原始轨迹，不做处理。"""
    return traj.copy()


def method_douglas_peucker(traj, tolerance):
    """Douglas-Peucker 简化。"""
    try:
        from shapely.geometry import LineString
        line = LineString(traj)
        simplified = line.simplify(tolerance, preserve_topology=True)
        return np.array(simplified.coords)
    except ImportError:
        # 回退：简单的基于距离的简化
        if len(traj) <= 2:
            return traj.copy()
        keep = [0]
        _dp_recursive(traj, 0, len(traj) - 1, tolerance, keep)
        keep.append(len(traj) - 1)
        keep.sort()
        return traj[keep]


def _dp_recursive(points, start, end, tol, keep):
    if end - start <= 1:
        return
    max_dist = 0
    max_idx = start
    p1, p2 = points[start], points[end]
    for i in range(start + 1, end):
        d = _point_line_dist(points[i], p1, p2)
        if d > max_dist:
            max_dist = d
            max_idx = i
    if max_dist > tol:
        keep.append(max_idx)
        _dp_recursive(points, start, max_idx, tol, keep)
        _dp_recursive(points, max_idx, end, tol, keep)


def _point_line_dist(p, a, b):
    ab = b - a
    ap = p - a
    ab_len = np.linalg.norm(ab)
    if ab_len < 1e-8:
        return np.linalg.norm(ap)
    t = np.clip(np.dot(ap, ab) / (ab_len ** 2), 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def method_bspline(traj, n_points=100):
    """B-spline 拟合。"""
    try:
        from scipy.interpolate import splprep, splev
        if len(traj) < 4:
            return traj.copy()
        tck, u = splprep([traj[:, 0], traj[:, 1]], s=0.5, k=min(3, len(traj) - 1))
        u_new = np.linspace(0, 1, n_points)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new])
    except Exception:
        return traj.copy()


def method_constrained(traj, env_map, map_size, cell_size=1.0, max_iter=10):
    """带坡度约束的 B-spline。"""
    try:
        from scipy.interpolate import splprep, splev
    except ImportError:
        return traj.copy()

    if len(traj) < 4:
        return traj.copy()

    current = traj.copy()
    fitted = None
    for iteration in range(max_iter):
        if len(current) < 4:
            break
        # 去除连续重复点，splprep 要求相邻点不重合
        diffs = np.diff(current, axis=0)
        mask = np.concatenate([[True], np.any(np.abs(diffs) > 1e-8, axis=1)])
        unique_pts = current[mask]
        if len(unique_pts) < 4:
            break
        try:
            tck, u = splprep([unique_pts[:, 0], unique_pts[:, 1]],
                             s=max(0.3, 0.5 - iteration * 0.02),
                             k=min(3, len(unique_pts) - 1))
        except (ValueError, TypeError):
            break
        u_new = np.linspace(0, 1, 100)
        x_new, y_new = splev(u_new, tck)
        fitted = np.column_stack([x_new, y_new])

        # 检查坡度违规
        violations = []
        for i in range(len(fitted) - 1):
            slope = compute_slope(env_map, fitted[i], fitted[i + 1], map_size, cell_size)
            if slope > ADA_THRESHOLD:
                violations.append(i)

        if not violations:
            return fitted

        # 在违规段中点插入控制点
        new_points = list(current)
        for vi in violations[:5]:  # 每次最多修正 5 个
            mid = (fitted[vi] + fitted[vi + 1]) / 2
            # 微调：沿法线方向偏移
            tangent = fitted[min(vi + 1, len(fitted) - 1)] - fitted[vi]
            normal = np.array([-tangent[1], tangent[0]])
            n_len = np.linalg.norm(normal)
            if n_len > 1e-8:
                normal = normal / n_len
            offset = mid + normal * 0.5
            new_points.append(offset)
        current = np.array(new_points)
        # 按沿线距离排序
        dists = np.sqrt(np.sum((current - current[0]) ** 2, axis=1))
        order = np.argsort(dists)
        current = current[order]

    return fitted if fitted is not None else traj.copy()


def method_moving_avg(traj, window=3):
    """移动平均平滑。"""
    try:
        from scipy.ndimage import uniform_filter1d
        smoothed = np.column_stack([
            uniform_filter1d(traj[:, 0], size=window),
            uniform_filter1d(traj[:, 1], size=window),
        ])
        return smoothed
    except ImportError:
        if len(traj) < window:
            return traj.copy()
        kernel = np.ones(window) / window
        return np.column_stack([
            np.convolve(traj[:, 0], kernel, mode='same'),
            np.convolve(traj[:, 1], kernel, mode='same'),
        ])


# ========== 指标计算 ==========

def compute_metrics(traj, env_map, map_size, cell_size=1.0):
    """计算轨迹指标。"""
    if len(traj) < 2:
        return {'path_length': 0, 'num_points': len(traj),
                'max_slope': 0, 'mean_slope': 0, 'ada_violations': 0, 'ada_compliance_pct': 100}

    path_length = sum(np.linalg.norm(traj[i + 1] - traj[i]) for i in range(len(traj) - 1))
    slopes = [compute_slope(env_map, traj[i], traj[i + 1], map_size, cell_size)
              for i in range(len(traj) - 1)]
    violations = sum(1 for s in slopes if s > ADA_THRESHOLD)

    return {
        'path_length': float(path_length),
        'num_points': len(traj),
        'max_slope': float(max(slopes)) if slopes else 0,
        'mean_slope': float(np.mean(slopes)) if slopes else 0,
        'ada_violations': violations,
        'ada_compliance_pct': (1 - violations / max(len(slopes), 1)) * 100,
    }


# ========== 主流程 ==========

METHODS = {
    'Raw': lambda t, **kw: method_raw(t),
    'DP(tol=0.5)': lambda t, **kw: method_douglas_peucker(t, 0.5),
    'DP(tol=1.0)': lambda t, **kw: method_douglas_peucker(t, 1.0),
    'DP(tol=2.0)': lambda t, **kw: method_douglas_peucker(t, 2.0),
    'B-spline': lambda t, **kw: method_bspline(t),
    'Constrained': lambda t, **kw: method_constrained(t, kw['env_map'], kw['map_size']),
    'MovAvg': lambda t, **kw: method_moving_avg(t),
}


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
    base_env = env.env

    all_data_rows = []
    time_rows = []
    geojson_features = []
    best_ep_data = None
    best_reward = -float('inf')

    for ep in range(args.episodes):
        # 环境重置计时
        t0 = time.perf_counter()
        obs_n = env.reset()
        t_env_reset = time.perf_counter() - t0

        # RL 推理
        done = False
        steps = 0
        ep_return = 0.0
        t_rl_start = time.perf_counter()

        positions_history = [[] for _ in range(base_env.num_agents)]
        for i in range(base_env.num_agents):
            positions_history[i].append(base_env.agent_positions[i].copy())

        while (not done) and steps < MAX_STEPS:
            steps += 1
            action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
            obs_n, reward_n, done_n, _ = env.step(action_n)
            done = all(done_n)
            ep_return += float(np.sum(reward_n))
            for i in range(base_env.num_agents):
                positions_history[i].append(base_env.agent_positions[i].copy())

        t_rl_inference = time.perf_counter() - t_rl_start

        # 对每个 agent 的轨迹应用后处理
        env_map = base_env.map.copy()
        map_size = base_env.map_size
        method_times = {}

        for agent_idx in range(base_env.num_agents):
            raw_traj = np.array(positions_history[agent_idx])

            for method_name, method_fn in METHODS.items():
                t_start = time.perf_counter()
                processed = method_fn(raw_traj, env_map=env_map, map_size=map_size)
                t_elapsed = time.perf_counter() - t_start

                if method_name not in method_times:
                    method_times[method_name] = 0
                method_times[method_name] += t_elapsed

                metrics = compute_metrics(processed, env_map, map_size)
                metrics.update({
                    'episode': ep,
                    'agent': agent_idx,
                    'method': method_name,
                    'processing_time': t_elapsed,
                })
                all_data_rows.append(metrics)

        # 计时
        time_rows.append({
            'episode': ep,
            't_env_reset': t_env_reset,
            't_rl_inference': t_rl_inference,
            **{f't_{k}': v for k, v in method_times.items()},
        })

        if ep_return > best_reward:
            best_reward = ep_return
            best_ep_data = {
                'positions': positions_history,
                'env_map': env_map,
                'map_size': map_size,
            }

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}/{args.episodes}")

    # ========== 保存结果 ==========

    # 主数据 CSV
    csv_path = os.path.join(args.output_dir, 'postprocess_data.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_data_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_data_rows)
    print(f"Saved {csv_path}")

    # 计时 CSV
    time_csv = os.path.join(args.output_dir, 'time_breakdown.csv')
    with open(time_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(time_rows[0].keys()))
        writer.writeheader()
        writer.writerows(time_rows)
    print(f"Saved {time_csv}")

    # 汇总 CSV
    import pandas as pd
    df = pd.DataFrame(all_data_rows)
    summary = df.groupby('method').agg({
        'path_length': ['mean', 'std'],
        'max_slope': ['mean', 'std'],
        'ada_compliance_pct': ['mean', 'std'],
        'num_points': 'mean',
        'processing_time': 'mean',
    }).round(3)
    summary.to_csv(os.path.join(args.output_dir, 'postprocess_summary.csv'))
    print("Saved postprocess_summary.csv")

    # GeoJSON 导出
    if best_ep_data:
        features = []
        for i in range(3):
            raw = np.array(best_ep_data['positions'][i])
            coords = [[float(p[0]), float(p[1])] for p in raw]
            features.append({
                "type": "Feature",
                "properties": {"agent": i, "method": "raw"},
                "geometry": {"type": "LineString", "coordinates": coords}
            })
        geojson = {"type": "FeatureCollection", "features": features}
        geo_path = os.path.join(args.output_dir, 'trajectories.geojson')
        with open(geo_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"Saved {geo_path}")

    # ========== 可视化 ==========

    method_names = list(METHODS.keys())
    metric_keys = ['path_length', 'max_slope', 'ada_compliance_pct', 'num_points', 'processing_time']
    metric_labels = ['Path Length', 'Max Slope (%)', 'ADA Compliance (%)', 'Num Points', 'Processing Time (s)']

    # 方法对比柱状图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for mi, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        if mi >= len(axes):
            break
        ax = axes[mi]
        means_m = [df[df['method'] == m][mk].mean() for m in method_names]
        stds_m = [df[df['method'] == m][mk].std() for m in method_names]
        x = np.arange(len(method_names))
        colors = [nature_style.COLOR_LIST[i % len(nature_style.COLOR_LIST)] for i in range(len(method_names))]
        ax.bar(x, means_m, yerr=stds_m, capsize=3, color=colors, edgecolor='white', width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=30, ha='right', fontsize=8)
        ax.set_title(ml)

    if len(axes) > len(metric_keys):
        axes[-1].set_visible(False)
    fig.suptitle('Post-Processing Method Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'postprocess_comparison.png'))
    plt.close(fig)
    print("Saved postprocess_comparison.png")

    # 最佳 episode 轨迹对比
    if best_ep_data:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        agent_traj = np.array(best_ep_data['positions'][0])
        env_map = best_ep_data['env_map']
        map_size = best_ep_data['map_size']

        for mi, (mname, mfn) in enumerate(list(METHODS.items())[:6]):
            ax = axes[mi]
            ax.imshow(env_map[1], cmap='terrain', alpha=0.4, origin='lower')
            processed = mfn(agent_traj, env_map=env_map, map_size=map_size)
            ax.plot(agent_traj[:, 0], agent_traj[:, 1], 'k--', alpha=0.3, label='Raw')
            ax.plot(processed[:, 0], processed[:, 1],
                    color=nature_style.COLOR_LIST[mi % len(nature_style.COLOR_LIST)],
                    linewidth=2, label=mname)
            ax.set_title(mname, fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xlim(0, map_size)
            ax.set_ylim(0, map_size)

        fig.suptitle('Trajectory Comparison (Agent 0, Best Episode)', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, 'postprocess_trajectory.png'))
        plt.close(fig)
        print("Saved postprocess_trajectory.png")

    # 坡度剖面对比
    if best_ep_data:
        agent_traj = np.array(best_ep_data['positions'][0])
        raw_slopes = [compute_slope(env_map, agent_traj[i], agent_traj[i + 1], map_size)
                      for i in range(len(agent_traj) - 1)]
        constrained = method_constrained(agent_traj, env_map, map_size)
        con_slopes = [compute_slope(env_map, constrained[i], constrained[i + 1], map_size)
                      for i in range(len(constrained) - 1)]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(raw_slopes)), raw_slopes, color=nature_style.COLOR_LIST[0],
                label='Raw', linewidth=1.5)
        ax.plot(np.linspace(0, len(raw_slopes), len(con_slopes)), con_slopes,
                color=nature_style.COLOR_LIST[2], label='Constrained', linewidth=1.5)
        ax.axhline(y=ADA_THRESHOLD, color='red', linestyle='--', alpha=0.5, label=f'ADA Threshold ({ADA_THRESHOLD}%)')
        ax.set_xlabel('Segment Index')
        ax.set_ylabel('Slope (%)')
        ax.set_title('Slope Profile Comparison')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, 'slope_profile.png'))
        plt.close(fig)
        print("Saved slope_profile.png")

    # 计时堆叠图
    time_df = pd.DataFrame(time_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    time_cols = [c for c in time_df.columns if c.startswith('t_') and c != 't_env_reset']
    mean_times = [time_df[c].mean() for c in ['t_env_reset', 't_rl_inference'] + [f't_{m}' for m in method_names if f't_{m}' in time_df.columns]]
    time_labels = ['Env Reset', 'RL Inference'] + [m for m in method_names if f't_{m}' in time_df.columns]

    axes[0].bar(range(len(mean_times)), mean_times,
                color=[nature_style.COLOR_LIST[i % len(nature_style.COLOR_LIST)] for i in range(len(mean_times))])
    axes[0].set_xticks(range(len(time_labels)))
    axes[0].set_xticklabels(time_labels, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('Time (s)')
    axes[0].set_title('Mean Time per Episode')

    # 饼图
    axes[1].pie(mean_times[:4], labels=time_labels[:4], autopct='%1.1f%%',
                colors=nature_style.COLOR_LIST[:4])
    axes[1].set_title('Time Distribution')

    fig.suptitle('Time Breakdown', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'time_breakdown.png'))
    plt.close(fig)
    print("Saved time_breakdown.png")


if __name__ == '__main__':
    main()
