"""
独立 DDPG 训练。

关键区别: Critic 只输入自己的 obs+act (184维)，不是所有 agent 的 (552维)。

运行:
    cd examples/MADDPG_18_hyper_vision
    CUDA_VISIBLE_DEVICES=0 python train_independent_ddpg.py \
        --max_episodes 1000000 \
        --model_dir ./model_independent_ddpg
"""
import os
import sys
import csv
import argparse
import numpy as np

import paddle
import paddle.nn as nn
import parl
from parl.algorithms import MADDPG
from parl.utils import logger, summary
from maddpg.envs import ParkEnv
from maddpg.envs import MADDPGWrapper
from maddpg.agents import MAAgent
from maddpg.models.hyper_model import ActorModel

paddle.set_device('gpu')

CRITIC_LR = 0.001
ACTOR_LR = 0.0001
GAMMA = 0.95
TAU = 0.001
BATCH_SIZE = 512
MAX_STEP_PER_EPISODE = 120
EVAL_EPISODES = 3


class IndependentCriticModel(parl.Model):
    """Critic 只看自己的 obs + act（独立，非集中式）。"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        input_dim = obs_dim + act_dim  # 180 + 4 = 184
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs, act):
        x = paddle.concat([obs, act], axis=-1)
        x = paddle.nn.functional.relu(self.fc1(x))
        x = paddle.nn.functional.relu(self.fc2(x))
        return self.fc3(x)


ELEVATION_OBS_SIZE = 11


class IndependentMAModel(parl.Model):
    """使用独立 Critic 的 MAModel。"""
    def __init__(self, obs_dim, act_dim, obs_shape_n, act_shape_n, continuous_actions=True):
        super().__init__()
        obs_flat = obs_dim[0] if isinstance(obs_dim, (tuple, list)) else obs_dim
        act_flat = act_dim if isinstance(act_dim, int) else int(np.prod(act_dim))

        self.actor_model = ActorModel(obs_flat, act_flat, continuous_actions=True)
        self.critic_model = IndependentCriticModel(obs_flat, act_flat)
        self._obs_flat = obs_flat
        self._act_flat = act_flat

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, act):
        # MADDPG.Q() 传入所有 agent 的 obs 和 act 拼接，我们只取自己的部分
        own_obs = obs[:, :self._obs_flat]
        own_act = act[:, :self._act_flat]
        return self.critic_model(own_obs, own_act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


def run_episode(env, agents):
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
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i],
                                 next_obs_n[i], done_n[i])
        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward
        for i, agent in enumerate(agents):
            agent.learn(agents)
    return total_reward, agents_reward, steps


def run_evaluate_episodes(env, agents, eval_episodes):
    eval_rewards = []
    for _ in range(eval_episodes):
        obs_n = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < MAX_STEP_PER_EPISODE:
            steps += 1
            action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
            obs_n, reward_n, done_n, _ = env.step(action_n)
            done = all(done_n)
            total_reward += sum(reward_n)
        eval_rewards.append(total_reward)
    return eval_rewards


def main():
    logger.set_dir('./train_log/independent_ddpg')

    env = ParkEnv(num_agents=3, render_mode=False)
    wrapped_env = MADDPGWrapper(env, continuous_actions=True)
    env = wrapped_env

    act_space_n = [env.action_space[f'agent_{i}'] for i in range(env.n)]

    agents = []
    for i in range(env.n):
        model = IndependentMAModel(
            obs_dim=env.obs_shape_n[i],
            act_dim=env.act_shape_n[i],
            obs_shape_n=env.obs_shape_n,
            act_shape_n=env.act_shape_n,
            continuous_actions=True
        )
        algorithm = MADDPG(
            model, agent_index=i, act_space=act_space_n,
            gamma=GAMMA, tau=TAU, critic_lr=CRITIC_LR, actor_lr=ACTOR_LR
        )
        agent = MAAgent(
            algorithm, agent_index=i,
            obs_dim_n=env.obs_shape_n, act_dim_n=env.act_shape_n,
            batch_size=BATCH_SIZE
        )
        agents.append(agent)

    if args.restore:
        for i in range(len(agents)):
            model_file = os.path.join(args.model_dir, f'agent_{i}')
            if os.path.exists(model_file):
                agents[i].restore(model_file)

    os.makedirs(args.model_dir, exist_ok=True)
    csv_path = args.model_dir + '_training.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'reward', 'agent0_reward', 'agent1_reward', 'agent2_reward', 'steps'])

    total_steps = 0
    total_episodes = 0
    while total_episodes <= args.max_episodes:
        ep_reward, ep_agent_rewards, steps = run_episode(env, agents)
        csv_writer.writerow([total_episodes, ep_reward] + list(ep_agent_rewards) + [steps])
        if total_episodes % 100 == 0:
            csv_file.flush()
            logger.info(f'episode {total_episodes}, reward {ep_reward:.1f}, steps {steps}')

        total_steps += steps
        total_episodes += 1

        if total_episodes % args.test_every_episodes == 0:
            eval_rewards = run_evaluate_episodes(env, agents, EVAL_EPISODES)
            logger.info(f'Eval over {EVAL_EPISODES} episodes, Reward: {np.mean(eval_rewards):.1f}')
            for i in range(len(agents)):
                agents[i].save(os.path.join(args.model_dir, f'agent_{i}'))

    csv_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default=1000000)
    parser.add_argument('--model_dir', type=str, default='./model_independent_ddpg')
    parser.add_argument('--test_every_episodes', type=int, default=1000)
    parser.add_argument('--restore', action='store_true', default=False)
    args = parser.parse_args()
    main()
