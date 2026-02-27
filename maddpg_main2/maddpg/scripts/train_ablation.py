import os
import csv
import time
from datetime import datetime
import argparse
import numpy as np
# Model import is deferred to main() based on --model_type flag
from maddpg.agents import MAAgent
from parl.algorithms import MADDPG
from parl.utils import logger, summary
from gym import spaces
from maddpg.envs import ParkEnv
from maddpg.envs import MADDPGWrapper
import paddle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 设置设备为 GPU
paddle.set_device('gpu')

CRITIC_LR = 0.001  # learning rate for the critic model
ACTOR_LR = 0.0001  # learning rate of the actor model
GAMMA = 0.95  # reward discount factor
TAU = 0.001  # soft update
BATCH_SIZE = 512
MAX_STEP_PER_EPISODE =120  # maximum step per episode #原来是25
EVAL_EPISODES = 3


# Runs policy and returns episodes' rewards and steps for evaluation
def run_evaluate_episodes(env, agents, eval_episodes):
    eval_episode_rewards = []
    eval_episode_steps = []
    while len(eval_episode_rewards) < eval_episodes:
        obs_n = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < MAX_STEP_PER_EPISODE:
            steps += 1
            action_n = [
                agent.predict(obs) for agent, obs in zip(agents, obs_n)
            ]
            obs_n, reward_n, done_n, _ = env.step(action_n)
            done = all(done_n)
            total_reward += sum(reward_n)
            # show animation
            if args.show:
                time.sleep(0.1)
                env.render()

        eval_episode_rewards.append(total_reward)
        eval_episode_steps.append(steps)
    return eval_episode_rewards, eval_episode_steps


def run_episode(env, agents):
    obs_n = env.reset()
    done = False
    total_reward = 0
    agents_reward = [0 for _ in range(env.n)]
    steps = 0
    losses = []  # 用于存储所有智能体的损失值
    while not done and steps < MAX_STEP_PER_EPISODE:
        steps += 1
        action_n = [agent.sample(obs) for agent, obs in zip(agents, obs_n)]
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)

        # if args.show:
        #     env.render()

        # store experience
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i],
                                 next_obs_n[i], done_n[i])

        # compute reward of every agent
        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward

        # show model effect without training
        if args.restore and args.show:
            continue

        # learn policy
        for i, agent in enumerate(agents):
            critic_loss = agent.learn(agents)
            if critic_loss is not None:
                losses.append(critic_loss)  # critic_loss是一个浮点数的对象



    # 计算平均损失值
    average_loss = np.mean(losses) if losses else 0.0

    return total_reward, agents_reward, steps, average_loss


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


def main():
    MAModel = _import_model(args.model_type)
    logger.info(f"Using model type: {args.model_type}")

    # 将时间戳添加到日志目录的后缀中，防止被替换
    logger.set_dir('./train_log/{}_{}'.format(args.env, args.continuous_actions))


    # 创建环境实例
    env = ParkEnv(num_agents=3, render_mode=False,
            use_elevation_obs=(not args.no_elevation_obs),
            use_slope_penalty=(not args.no_slope_penalty),
            flat_terrain=args.flat_terrain,
            cell_size_m=args.cell_size_m,
            use_poi_reward=(not args.no_poi_reward),
            use_trajectory_penalty=(not args.no_trajectory_penalty))
    wrapped_env = MADDPGWrapper(env, continuous_actions=True)
    env = wrapped_env

    # 打印 action_space 的类型和内容
    print(f"Type of env.action_space: {type(env.action_space)}")
    print(f"env.action_space keys: {list(env.action_space.keys())}")

    if args.continuous_actions:
        assert isinstance(env.action_space['agent_0'], spaces.Box), "Action space should be Box for continuous actions"

    # 计算观测空间展平后的尺寸
    obs_size_n = [int(np.prod(shape)) for shape in env.obs_shape_n]
    act_dim_n = env.act_shape_n  # [2, 2, 2]
    critic_in_dim = sum(obs_size_n) + sum(act_dim_n)
    print(f"env.obs_shape_n: {env.obs_shape_n}, env.act_shape_n: {env.act_shape_n}")
    print(f"obs_size_n: {obs_size_n}")
    print(f"critic_in_dim: {critic_in_dim}")

    # 构建所有智能体的动作空间列表
    act_space_n = [env.action_space[f'agent_{i}'] for i in range(env.n)]

    # 构建智能体
    agents = []
    for i in range(env.n):
        model = MAModel(
            obs_dim=env.obs_shape_n[i],      # 单个智能体的观测空间形状
            act_dim=env.act_shape_n[i],      # 单个智能体的动作空间维度
            obs_shape_n=env.obs_shape_n,     # 所有智能体的观测空间形状列表
            act_shape_n=env.act_shape_n,     # 所有智能体的动作空间维度列表
            continuous_actions=args.continuous_actions
        )
        algorithm = MADDPG(
            model,
            agent_index=i,
            act_space=act_space_n,  # 传递所有智能体的动作空间列表
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR
        )
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=env.obs_shape_n,  # 包含所有智能体观测空间形状的列表
            act_dim_n=env.act_shape_n,  # 包含所有智能体动作空间维度的列表
            batch_size=BATCH_SIZE
        )
        agents.append(agent)


    if args.restore:
        # restore model
        for i in range(len(agents)):
            model_file = args.model_dir + '/agent_' + str(i)
            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} does not exits'.format(model_file))
            agents[i].restore(model_file)

    # CSV 日志
    csv_dir = os.path.dirname(args.model_dir) if os.path.dirname(args.model_dir) else '.'
    csv_path = os.path.join(csv_dir, os.path.basename(args.model_dir) + '_training.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'reward', 'agent0_reward', 'agent1_reward', 'agent2_reward', 'steps', 'loss'])
    logger.info(f"CSV log: {csv_path}")

    total_steps = 0
    total_episodes = 0
    while total_episodes <= args.max_episodes:
        # run an episode
        ep_reward, ep_agent_rewards, steps, ep_loss = run_episode(env, agents)

        # 在每个回合结束后，如果需要，渲染环境
        if args.show:
            env.render()

        # 记录和打印信息
        # summary.add_scalar('train/episode_reward_wrt_episode', ep_reward,
        #                    total_episodes)
        # summary.add_scalar('train/episode_reward_wrt_step', ep_reward,
        #                    total_steps)

        # 记录奖励值和损失值
        csv_writer.writerow([total_episodes, ep_reward] + list(ep_agent_rewards) + [steps, ep_loss])
        if total_episodes % 100 == 0:
            csv_file.flush()
            summary.add_scalar('train/episode_reward_wrt_episode', ep_reward, total_episodes)
            summary.add_scalar('train/episode_reward_wrt_step', ep_reward, total_steps)
            for i, agent_reward in enumerate(ep_agent_rewards):
                summary.add_scalar(f"Reward/Agent_{i}_Reward", agent_reward, total_episodes)
            summary.add_scalar('Loss/Critic_Loss', ep_loss, total_episodes)
            logger.info(
                'total_steps {}, episode {}, reward {}, agents rewards {}, episode steps {}'
                .format(total_steps, total_episodes, ep_reward, ep_agent_rewards,
                        steps))

        # 更新环境的步数和回合数
        env.total_steps = total_steps
        env.total_episodes = total_episodes

        total_steps += steps
        total_episodes += 1

        # evaluste agents
        if total_episodes % args.test_every_episodes == 0:

            eval_episode_rewards, eval_episode_steps = run_evaluate_episodes(
                env, agents, EVAL_EPISODES)
            summary.add_scalar('eval/episode_reward',
                               np.mean(eval_episode_rewards), total_episodes)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                EVAL_EPISODES, np.mean(eval_episode_rewards)))

            # save model
            model_dir = args.model_dir
            os.makedirs(model_dir, exist_ok=True)
            for i in range(len(agents)):
                model_name = '/agent_' + str(i)
                agents[i].save(model_dir + model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument(
        '--env',
        type=str,
        default='Park_Env',
        help='Park_Env designed by Zihuan Zhang')
    # auto save model, optional restore model
    parser.add_argument(
        '--show', action='store_true', default=False, help='display or not')
    parser.add_argument(
        '--restore',
        action='store_true',
        default=False,
        help='restore or not, must have model_dir')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./model',
        help='directory for saving model')
    parser.add_argument(
        '--model_type',
        type=str,
        default='hyper',
        choices=['hyper', 'mlp', 'unet', 'attention'],
        help='model architecture: hyper (CNN-MLP), mlp, unet, attention')
    parser.add_argument(
        '--continuous_actions',
        action='store_true',
        default=True,
        help='use continuous action mode or not')
        # Ablation switches (keep obs dim unchanged; disable terms by zeroing / skipping)
    parser.add_argument('--no_elevation_obs', action='store_true', default=False,
                        help='disable elevation patch in observation (zero-filled)')
    parser.add_argument('--no_slope_penalty', action='store_true', default=False,
                        help='disable slope/terrain penalty term in reward')
    parser.add_argument('--flat_terrain', action='store_true', default=False,
                        help='force flat terrain (all-zero elevation map)')
    parser.add_argument('--cell_size_m', type=float, default=1.0,
                        help='grid resolution in meters per cell (for evaluation/reporting)')
    parser.add_argument('--no_poi_reward', action='store_true', default=False,
                        help='disable POI proximity reward')
    parser.add_argument('--no_trajectory_penalty', action='store_true', default=False,
                        help='disable trajectory crossing and stuck penalties')
    parser.add_argument(
            '--max_episodes',
            type=int,
            default=2000000,
            help='stop condition: number of episodes')
    parser.add_argument(
        '--test_every_episodes',
        type=int,
        default=int(1e3),
        help='the episode interval between two consecutive evaluations')
    args = parser.parse_args()

    main()
