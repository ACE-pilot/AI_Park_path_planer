"""
共享评估工具函数。
所有评估脚本 import 此模块，避免重复代码。
"""
import os
import numpy as np
from maddpg.models.hyper_model import MAModel
from maddpg.agents import MAAgent
from parl.algorithms import MADDPG
from maddpg.envs import ParkEnv, MADDPGWrapper

CRITIC_LR = 0.001
ACTOR_LR = 0.0001
GAMMA = 0.95
TAU = 0.001
BATCH_SIZE = 1024
MAX_STEP_PER_EPISODE = 120


def setup_env():
    """创建并包装环境。"""
    env = ParkEnv(num_agents=3, render_mode=False)
    wrapped_env = MADDPGWrapper(env, continuous_actions=True)
    return wrapped_env


def build_agents(env):
    """构建 3 个 MADDPG agent（5 参数 MAModel）。"""
    act_space_n = [env.action_space[f'agent_{i}'] for i in range(env.n)]
    agents = []
    for i in range(env.n):
        model = MAModel(
            obs_dim=env.obs_shape_n[i],
            act_dim=env.act_shape_n[i],
            obs_shape_n=env.obs_shape_n,
            act_shape_n=env.act_shape_n,
            continuous_actions=True
        )
        algorithm = MADDPG(
            model,
            agent_index=i,
            act_space=act_space_n,
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR
        )
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=env.obs_shape_n,
            act_dim_n=env.act_shape_n,
            batch_size=BATCH_SIZE
        )
        agents.append(agent)
    return agents


def restore_agents(agents, model_dir):
    """从 model_dir/agent_0, agent_1, agent_2 加载检查点。"""
    for i, agent in enumerate(agents):
        model_file = os.path.join(model_dir, f"agent_{i}")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        agent.restore(model_file)


def run_eval_episode(env, agents, max_steps=120):
    """跑一个评估 episode，返回指标 dict。"""
    obs_n = env.reset()
    base_env = env.env
    done = False
    steps = 0
    ep_return = 0.0

    prev_positions = [base_env.agent_positions[i].copy() for i in range(base_env.num_agents)]
    path_len = [0.0] * base_env.num_agents
    cum_abs_dh = [0.0] * base_env.num_agents
    max_abs_dh = [0.0] * base_env.num_agents
    positions_history = [[] for _ in range(base_env.num_agents)]

    for i in range(base_env.num_agents):
        positions_history[i].append(base_env.agent_positions[i].copy())

    while (not done) and steps < max_steps:
        steps += 1
        action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)
        ep_return += float(np.sum(reward_n))

        for i in range(base_env.num_agents):
            pos = base_env.agent_positions[i].copy()
            positions_history[i].append(pos)
            d = float(np.linalg.norm(pos - prev_positions[i]))
            path_len[i] += d

            x_old = int(np.clip(round(prev_positions[i][0]), 0, base_env.map_size - 1))
            y_old = int(np.clip(round(prev_positions[i][1]), 0, base_env.map_size - 1))
            x_new = int(np.clip(round(pos[0]), 0, base_env.map_size - 1))
            y_new = int(np.clip(round(pos[1]), 0, base_env.map_size - 1))

            dh = float(int(base_env.map[1, y_new, x_new]) - int(base_env.map[1, y_old, x_old]))
            abs_dh = abs(dh)
            cum_abs_dh[i] += abs_dh
            if abs_dh > max_abs_dh[i]:
                max_abs_dh[i] = abs_dh
            prev_positions[i] = pos

        obs_n = next_obs_n

    cell_size = getattr(base_env, "cell_size_m", 1.0)
    return {
        "success": 1 if done else 0,
        "episode_return": float(ep_return),
        "steps": int(steps),
        "path_len_mean": float(np.mean(path_len)),
        "max_slope_proxy_mean": float(np.mean([m / max(cell_size, 1e-6) for m in max_abs_dh])),
        "earthwork_proxy_mean": float(np.mean(cum_abs_dh)),
        "positions_history": positions_history,
        "path_len": path_len,
    }


def get_elevation(env_map, pos, map_size):
    """查询环境地图上某位置的高程值。"""
    x = int(np.clip(round(pos[0]), 0, map_size - 1))
    y = int(np.clip(round(pos[1]), 0, map_size - 1))
    return float(env_map[1, y, x])


def compute_slope(env_map, p1, p2, map_size, cell_size=1.0):
    """计算两点之间的坡度百分比。"""
    e1 = get_elevation(env_map, p1, map_size)
    e2 = get_elevation(env_map, p2, map_size)
    horiz = float(np.linalg.norm(np.array(p2) - np.array(p1))) * cell_size
    if horiz < 1e-6:
        return 0.0
    return abs(e2 - e1) / horiz * 100.0
