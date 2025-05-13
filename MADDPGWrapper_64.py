import gym
from gym import spaces
import numpy as np
import gym
from gym import spaces

class MADDPGWrapper(gym.Wrapper):
    def __init__(self, env=None, continuous_actions=False):
        super(MADDPGWrapper, self).__init__(env)
        self.env = env
        self.continuous_actions = continuous_actions
        
        map_size = 64
        observation_shape = (3, map_size, map_size)  # 保持为 3 个通道
        single_observation_space = spaces.Box(low=0, high=255, shape=observation_shape, dtype=np.uint8)
        observation_spaces = {agent: single_observation_space for agent in self.agents}
 
        self.observation_spaces = observation_spaces  # dict
        self.action_spaces = self.env.action_spaces  # dict

        # 将观察空间和动作空间转换为列表
        self.obs_shape_n = [space.shape for space in self.observation_spaces.values()]
        self.act_shape_n = [self.get_shape(space) for space in self.action_spaces.values()]
        self.n = len(self.obs_shape_n)  # 智能体数量
        print(f"Number of agents (self.n): {self.n}")

        # 将 action_spaces 赋值给 action_space
        self.action_space = self.action_spaces

        # 定义 agents_name
        self.agents_name = [f'agent_{i}' for i in range(self.n)]

    def get_shape(self, input_space):
        if isinstance(input_space, spaces.Box):
            return input_space.shape[0]  # 假设动作空间是一维的
        elif isinstance(input_space, spaces.Discrete):
            return input_space.n
        else:
            raise NotImplementedError(f"Shape not implemented for {type(input_space)}")

    def step(self, action_n):
        # 将动作列表转换为动作字典
        action_dict = {}
        for i, action in enumerate(action_n):
            agent_name = self.agents_name[i]
            action_space = self.action_space[agent_name]
            # 确保动作在动作空间内
            if isinstance(action_space, spaces.Discrete):
                action = int(action)
            elif isinstance(action_space, spaces.Box):
                action = np.clip(action, action_space.low, action_space.high)
            else:
                raise NotImplementedError(f"Action space type {type(action_space)} not supported")
            action_dict[agent_name] = action

        # 与环境交互
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)

        # 将字典转换为列表
        obs_n = [obs_dict[agent_name] for agent_name in self.agents_name]
        reward_n = [reward_dict[agent_name] for agent_name in self.agents_name]
        done_n = [done_dict[agent_name] for agent_name in self.agents_name]
        info_n = [info_dict.get(agent_name, {}) for agent_name in self.agents_name]

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_dict = self.env.reset()
        obs_n = [obs_dict[agent_name] for agent_name in self.agents_name]
        return obs_n
