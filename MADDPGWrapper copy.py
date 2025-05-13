import gym
from gym import spaces
import numpy as np

class MADDPGWrapper(gym.Wrapper):
    def __init__(self, env=None, continuous_actions=False):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.continuous_actions = continuous_actions
        # 从环境中获取 observation_spaces 和 action_spaces，它们是字典
        self.observation_spaces = self.env.observation_spaces  # 字典
        self.action_spaces = self.env.action_spaces  # 字典
        # 将字典的值（空间对象）转换为列表
        self.observation_space = list(self.observation_spaces.values())
        self.action_space = list(self.action_spaces.values())
        assert len(self.observation_space) == len(self.action_space)
        self.n = len(self.observation_space)
        # 获取智能体名称列表
        self.agents_name = list(self.observation_spaces.keys())
        self.obs_shape_n = [
            self.get_shape(self.observation_space[i]) for i in range(self.n)
        ]
        self.act_shape_n = [
            self.get_shape(self.action_space[i]) for i in range(self.n)
        ]

    def get_shape(self, input_space):
        if isinstance(input_space, spaces.Box):
            if len(input_space.shape) == 1:
                return input_space.shape[0]
            else:
                return input_space.shape
        elif isinstance(input_space, spaces.Discrete):
            return input_space.n
        else:
            print('[Error] shape is {}, not Box or Discrete'.format(
                input_space))
            raise NotImplementedError

    def reset(self):
        obs = self.env.reset()
        # 返回列表形式的观测
        return list(obs.values())

    def step(self, actions):
        actions_dict = {}
        for i, act in enumerate(actions):
            agent = self.agents_name[i]
            if self.continuous_actions:
                assert np.all(((act <= 1.0 + 1e-3), (act >= -1.0 - 1e-3))), \
                    'the action should be in range [-1.0, 1.0], but got {}'.format(act)
                high = self.action_space[i].high
                low = self.action_space[i].low
                mapped_action = low + (act - (-1.0)) * ((high - low) / 2.0)
                mapped_action = np.clip(mapped_action, low, high)
                actions_dict[agent] = mapped_action
            else:
                actions_dict[agent] = np.argmax(act)
        obs, rewards, dones, infos = self.env.step(actions_dict)
        return list(obs.values()), list(rewards.values()), list(
            dones.values()), list(infos.values())
