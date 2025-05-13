from park_envE import ParkEnv
from MADDPGWrapper import MADDPGWrapper
import numpy as np
import gym

# 创建环境实例
env = ParkEnv(num_agents=3, render_mode=False)
wrapped_env = MADDPGWrapper(env, continuous_actions=True)

num_episodes = 3  # 设置要运行的回合数

for episode in range(num_episodes):
    obs = wrapped_env.reset()
    done = [False] * wrapped_env.n
    episode_rewards = [0.0 for _ in range(wrapped_env.n)]  # 记录每个智能体的累计奖励
    step_count = 0

    while not all(done):
        # 修改此处，使用 agent 名称获取动作空间
        actions = []
        for agent in wrapped_env.agents:
            action_space = wrapped_env.action_spaces[agent]
            action = action_space.sample()
            actions.append(action)

        obs, rewards, done, info = wrapped_env.step(actions)
        # 累计每个智能体的奖励
        for i in range(wrapped_env.n):
            episode_rewards[i] += rewards[i]
        step_count += 1

    # 在每个回合的最后一步渲染一次
    env.render()

    # 打印每个回合各个智能体的奖励和总的奖励
    total_reward = sum(episode_rewards)
    print(f"Episode {episode + 1}:")
    for i in range(wrapped_env.n):
        print(f"  Agent {i + 1} Reward: {episode_rewards[i]}")
    print(f"  Total Reward: {total_reward}\n")
