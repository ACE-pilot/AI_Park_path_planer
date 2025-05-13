import numpy as np
from park_env import ParkEnv
# 创建环境
env = ParkEnv(num_agents=1, render_mode=True)

# 重置环境
obs = env.reset()

done = [False]
total_reward = 0

while not all(done):
    # 为每个智能体选择随机连续动作，范围在 [-1, 1]
    actions = [env.action_space.sample() for _ in range(env.num_agents)]
    obs, rewards, done, info = env.step(actions)
    total_reward += sum(rewards)

print(f"Total Reward: {total_reward}")
env.close()
