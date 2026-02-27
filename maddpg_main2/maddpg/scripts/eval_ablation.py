import os
import argparse
import numpy as np

from maddpg.agents import MAAgent
from parl.algorithms import MADDPG
from parl.utils import logger

from maddpg.envs import ParkEnv
from maddpg.envs import MADDPGWrapper
import paddle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

CRITIC_LR = 0.001
ACTOR_LR = 0.0001
GAMMA = 0.95
TAU = 0.001
BATCH_SIZE = 1024


def _import_model(model_type):
    """Dynamically import MAModel based on --model_type flag."""
    if model_type == 'hyper':
        from maddpg.models.hyper_model import MAModel
    elif model_type == 'mlp':
        from maddpg.models.mlp_model import MAModel
    elif model_type == 'unet':
        from maddpg.models.unet_model import MAModel
    elif model_type == 'attention':
        from maddpg.models.attention_model import MAModel
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return MAModel


def build_agents(env, model_type='hyper'):
    # env is wrapped env (MADDPGWrapper)
    MAModel = _import_model(model_type)
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
        algo = MADDPG(
            model,
            agent_index=i,
            act_space=act_space_n,
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR
        )
        agent = MAAgent(
            algo,
            agent_index=i,
            obs_dim_n=env.obs_shape_n,
            act_dim_n=env.act_shape_n,
            batch_size=BATCH_SIZE
        )
        agents.append(agent)
    return agents

def restore_agents(agents, model_dir):
    for i, agent in enumerate(agents):
        model_file = os.path.join(model_dir, f"agent_{i}")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"model file not found: {model_file}")
        agent.restore(model_file)

def eval_one_episode(wrapped_env, agents, max_steps):
    obs_n = wrapped_env.reset()
    base_env = wrapped_env.env  # ParkEnv
    done = False
    steps = 0
    ep_return = 0.0

    # Metrics accumulators per agent
    prev_positions = [base_env.agent_positions[i].copy() for i in range(base_env.num_agents)]
    path_len = [0.0 for _ in range(base_env.num_agents)]
    cum_abs_dh = [0.0 for _ in range(base_env.num_agents)]
    max_abs_dh = [0.0 for _ in range(base_env.num_agents)]

    while (not done) and steps < max_steps:
        steps += 1
        action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
        next_obs_n, reward_n, done_n, _ = wrapped_env.step(action_n)
        done = all(done_n)

        ep_return += float(np.sum(reward_n))

        # Update metrics using base env positions and elevation map
        for i in range(base_env.num_agents):
            pos = base_env.agent_positions[i].copy()
            d = float(np.linalg.norm(pos - prev_positions[i]))
            path_len[i] += d

            x_old, y_old = int(round(prev_positions[i][0])), int(round(prev_positions[i][1]))
            x_new, y_new = int(round(pos[0])), int(round(pos[1]))
            x_old = int(np.clip(x_old, 0, base_env.map_size - 1))
            y_old = int(np.clip(y_old, 0, base_env.map_size - 1))
            x_new = int(np.clip(x_new, 0, base_env.map_size - 1))
            y_new = int(np.clip(y_new, 0, base_env.map_size - 1))

            dh = float(int(base_env.map[1, y_new, x_new]) - int(base_env.map[1, y_old, x_old]))
            abs_dh = abs(dh)
            cum_abs_dh[i] += abs_dh
            if abs_dh > max_abs_dh[i]:
                max_abs_dh[i] = abs_dh

            prev_positions[i] = pos

        obs_n = next_obs_n

    # Success: all agents done within max_steps
    success = 1 if done else 0

    # Aggregate to episode-level scalars (mean over agents)
    max_slope_proxy_mean = float(np.mean([m / max(getattr(base_env, "cell_size_m", 1.0), 1e-6) for m in max_abs_dh]))
    earthwork_proxy_mean = float(np.mean(cum_abs_dh))
    path_len_mean = float(np.mean(path_len))

    return {
        "success": success,
        "episode_return": float(ep_return),
        "steps": int(steps),
        "path_len_mean": path_len_mean,
        "max_slope_proxy_mean": max_slope_proxy_mean,
        "earthwork_proxy_mean": earthwork_proxy_mean,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="hyper",
                        choices=["hyper", "mlp", "unet", "attention"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--out_csv", type=str, default="ablation_eval.csv")

    # keep evaluation env consistent with training ablation switches
    parser.add_argument("--no_elevation_obs", action="store_true", default=False)
    parser.add_argument("--no_slope_penalty", action="store_true", default=False)
    parser.add_argument("--flat_terrain", action="store_true", default=False)
    parser.add_argument("--cell_size_m", type=float, default=1.0)
    parser.add_argument("--no_poi_reward", action="store_true", default=False)
    parser.add_argument("--no_trajectory_penalty", action="store_true", default=False)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Device
    paddle.set_device('gpu')

    env = ParkEnv(num_agents=3, render_mode=False,
                  use_elevation_obs=(not args.no_elevation_obs),
                  use_slope_penalty=(not args.no_slope_penalty),
                  flat_terrain=args.flat_terrain,
                  cell_size_m=args.cell_size_m,
                  use_poi_reward=(not args.no_poi_reward),
                  use_trajectory_penalty=(not args.no_trajectory_penalty))
    wrapped_env = MADDPGWrapper(env, continuous_actions=True)

    agents = build_agents(wrapped_env, model_type=args.model_type)
    restore_agents(agents, args.model_dir)

    rows = []
    for ep in range(args.episodes):
        res = eval_one_episode(wrapped_env, agents, args.max_steps)
        res.update({"config": args.config, "seed": args.seed, "episode": ep})
        rows.append(res)

    # append to CSV
    import csv
    file_exists = os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
