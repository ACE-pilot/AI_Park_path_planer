
# python test_CSV.py --restore --show


import os
import time
from datetime import datetime
import argparse
import numpy as np
#from simple_model import MAModel
from hyper_model import MAModel
from simple_agent import MAAgent
from parl.algorithms import MADDPG
from parl.utils import logger, summary
from gym import spaces
from park_envE1 import ParkEnv
from MADDPGWrapper import MADDPGWrapper
import paddle
import warnings
import csv

warnings.filterwarnings("ignore", category=DeprecationWarning)

# 设置设备为 GPU
paddle.set_device('gpu')

CRITIC_LR = 0.001  # learning rate for the critic model
ACTOR_LR = 0.0001  # learning rate of the actor model
GAMMA = 0.95       # reward discount factor
TAU = 0.001        # soft update
BATCH_SIZE = 1024
MAX_STEP_PER_EPISODE = 120  # maximum steps per episode
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

        # 存储经验
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i],
                                 next_obs_n[i], done_n[i])

        # 更新观测值并累计奖励
        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward
        
        # 若为纯演示模式则不进行学习
        if args.restore and args.show:
            continue

        # 学习策略
        for i, agent in enumerate(agents):
            critic_loss = agent.learn(agents)
            if critic_loss is not None:
                losses.append(critic_loss)

    average_loss = np.mean(losses) if losses else 0.0
    
    return total_reward, agents_reward, steps, average_loss


def main():
    # 将时间戳添加到日志目录的后缀中，防止被替换
    logger.set_dir('./train_log/{}_{}'.format(args.env, args.continuous_actions))

    # 创建环境实例并用 MADDPGWrapper 包装
    env_raw = ParkEnv(num_agents=3, render_mode=False)
    wrapped_env = MADDPGWrapper(env_raw, continuous_actions=True)
    env = wrapped_env

    # 打印 action_space 的类型和内容
    print(f"Type of env.action_space: {type(env.action_space)}")
    print(f"env.action_space keys: {list(env.action_space.keys())}")

    if args.continuous_actions:
        assert isinstance(env.action_space['agent_0'], spaces.Box), "Action space should be Box for continuous actions"

    # 计算展平后的观测空间尺寸和动作维度
    obs_size_n = [int(np.prod(shape)) for shape in env.obs_shape_n]
    act_dim_n = env.act_shape_n  # 例如 [2, 2, 2]
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
                    'model file {} does not exist'.format(model_file))
            agents[i].restore(model_file)

    total_steps = 0
    total_episodes = 0
    while total_episodes <= args.max_episodes:
        # 运行一个回合
        ep_reward, ep_agent_rewards, steps, ep_loss = run_episode(env, agents)
        
        # 如果达到一定奖励且为展示模式，则保存回合结果（SVG 和 CSV）
        if args.show and ep_reward > 2:
            save_dir = './savegoodresults/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # 保存 SVG 结果
            svg_save_path = os.path.join(save_dir, f'episode_{total_episodes}.svg')
            env.render(save_path=svg_save_path)
            # 同时保存 CSV 信息
            csv_file_path = os.path.join(save_dir, "episode_data.csv")
            csv_file_path1 = os.path.join(save_dir, "agent_steps.csv")
            env_raw.save_episode_csv(csv_file_path, episode_number=total_episodes)
            env_raw.save_agent_steps_csv(csv_file_path1, episode_number=total_episodes)

        # 记录和打印信息
        summary.add_scalar('train/episode_reward_wrt_episode', ep_reward, total_episodes)
        summary.add_scalar('train/episode_reward_wrt_step', ep_reward, total_steps)
        for i, agent_reward in enumerate(ep_agent_rewards):
            summary.add_scalar(f"Reward/Agent_{i}_Reward", agent_reward, total_episodes)
        summary.add_scalar('Loss/Critic_Loss', ep_loss, total_episodes)

        logger.info(
            'total_steps {}, episode {}, reward {}, agents rewards {}, episode steps {}'
            .format(total_steps, total_episodes, ep_reward, ep_agent_rewards, steps))

        # 更新环境内记录的步数和回合数
        env.total_steps = total_steps
        env.total_episodes = total_episodes

        total_steps += steps
        total_episodes += 1

        # 评估
        if total_episodes % args.test_every_episodes == 0:
            eval_episode_rewards, eval_episode_steps = run_evaluate_episodes(env, agents, EVAL_EPISODES)
            summary.add_scalar('eval/episode_reward', np.mean(eval_episode_rewards), total_episodes)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(EVAL_EPISODES, np.mean(eval_episode_rewards)))
            # 保存模型
            if not args.restore:
                model_dir = args.model_dir
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                for i in range(len(agents)):
                    model_name = '/agent_' + str(i)
                    agents[i].save(model_dir + model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment 参数
    parser.add_argument(
        '--env',
        type=str,
        default='Park_Env',
        help='Park_Env designed by Zihuan Zhang')
    # 是否显示或渲染环境
    parser.add_argument(
        '--show', action='store_true', default=False, help='display or not')
    # 是否恢复模型（使用训练好的模型）
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
        '--continuous_actions',
        action='store_true',
        default=True,
        help='use continuous action mode or not')
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
