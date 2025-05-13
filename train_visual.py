import os
import time
from datetime import datetime
import argparse
import numpy as np
#from simple_model import MAModel
from cnn_model_visual import MAModel, visualize_feature_maps
from simple_agent import MAAgent
from parl.algorithms import MADDPG
from parl.utils import logger, summary
from gym import spaces
from park_env import ParkEnv
from MADDPGWrapper import MADDPGWrapper
import paddle
import matplotlib.pyplot as plt

# 设置设备为 GPU
paddle.set_device('gpu')

CRITIC_LR = 0.01  # learning rate for the critic model
ACTOR_LR = 0.001  # learning rate of the actor model
GAMMA = 0.95  # reward discount factor
TAU = 0.01  # soft update
BATCH_SIZE = 1024
MAX_STEP_PER_EPISODE = 30  # maximum step per episode #原来是25
EVAL_EPISODES = 3
VISUALIZE_INTERVAL = 1  # 每隔多少回合进行一次特征图可视化



def run_episode(env, agents, episode_num):
    obs_n = env.reset()
    done = False
    total_reward = 0
    agents_reward = [0 for _ in range(env.n)]
    steps = 0
    while not done and steps < MAX_STEP_PER_EPISODE:
        steps += 1
        action_n = [agent.sample(obs) for agent, obs in zip(agents, obs_n)]
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)

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

    # 在指定的回合间隔进行特征图的可视化
    #if episode_num % VISUALIZE_INTERVAL == 0:
        # 这里假设使用第一个智能体的模型进行特征图可视化
        actor_model = agents[0].alg.model.actor_model
        obs = paddle.to_tensor(obs_n[0], dtype='float32').unsqueeze(0)  # 为了与模型输入匹配，添加 batch 维度
        means, act_std, conv1_out, conv2_out, conv3_out = actor_model.forward(obs)

        # 可视化卷积层的特征图
        visualize_feature_maps(conv1_out, 'Conv1')
        visualize_feature_maps(conv2_out, 'Conv2')
        visualize_feature_maps(conv3_out, 'Conv3')
        
        env.render()

    return total_reward, agents_reward, steps

def main():
    # 将时间戳添加到日志目录的后缀中，防止被替换
    logger.set_dir('./train_log/{}_{}'.format(args.env, args.continuous_actions))

    # 创建环境实例
    env = ParkEnv(num_agents=3, render_mode=False)
    wrapped_env = MADDPGWrapper(env, continuous_actions=True)
    env = wrapped_env

    # 打印 action_space 的类型和内容
    print(f"Type of env.action_space: {type(env.action_space)}")
    print(f"env.action_space keys: {list(env.action_space.keys())}")

    if args.continuous_actions:
        assert isinstance(env.action_space['agent_0'], spaces.Box), "Action space should be Box for continuous actions"

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
            act_space=[env.action_space[f'agent_{j}'] for j in range(env.n)],
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

    total_steps = 0
    total_episodes = 0
    while total_episodes <= args.max_episodes:
        # run an episode
        ep_reward, ep_agent_rewards, steps = run_episode(env, agents, total_episodes)
        
        # 在每个回合结束后，如果需要，渲染环境
        if args.show:
            env.render()

        # 记录和打印信息        
        summary.add_scalar('train/episode_reward_wrt_episode', ep_reward,
                           total_episodes)
        summary.add_scalar('train/episode_reward_wrt_step', ep_reward,
                           total_steps)
        logger.info(
            'total_steps {}, episode {}, reward {}, agents rewards {}, episode steps {}'
            .format(total_steps, total_episodes, ep_reward, ep_agent_rewards,
                    steps))

        total_steps += steps
        total_episodes += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--env', type=str, default='Park_Env', help='Park_Env designed by Zihuan Zhang')
    parser.add_argument('--show', action='store_true', default=False, help='display or not')
    parser.add_argument('--restore', action='store_true', default=False, help='restore or not, must have model_dir')
    parser.add_argument('--model_dir', type=str, default='./model', help='directory for saving model')
    parser.add_argument('--continuous_actions', action='store_true', default=True, help='use continuous action mode or not')
    parser.add_argument('--max_episodes', type=int, default=35000, help='stop condition: number of episodes')
    parser.add_argument('--test_every_episodes', type=int, default=int(1e3), help='the episode interval between two consecutive evaluations')
    args = parser.parse_args()

    main()
