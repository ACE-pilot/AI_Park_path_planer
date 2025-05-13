import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import random

class ParkEnv(gym.Env):
    def __init__(self, num_agents=1, render_mode=False):
        super(ParkEnv, self).__init__()
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.map_size = 128
        self.observation_shape = (3, self.map_size, self.map_size)
        
        # 将动作空间改为连续动作空间，速度在 [-1, 1] 之间的二维向量
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.observation_shape, dtype=np.uint8)

        # 初始化环境
        self._generate_map()
        self.agent_positions = []
        self.trajectories = [[] for _ in range(self.num_agents)]
        self.steps = 0
        self.max_steps = 500  # 最大步数，防止无限循环

    def _generate_map(self):
        # 生成地图
        self.map = np.zeros(self.observation_shape, dtype=np.uint8)
        # 通道 0：地块红线（0：外部，1：内部）
        self.map[0] = self._generate_boundary()
        # 通道 1：高差信息（0-255）
        self.map[1] = self._generate_elevation()
        # 通道 2：物体信息（0：无，其他值代表不同的物体）
        self.map[2] = self._generate_objects()

        # 设置起点和终点，确保在红线内且有一定的最小距离
        min_distance = self.map_size * 0.5  # 设定最小间距为地图大小的一半
        self.start_pos = self._get_random_position().astype(np.float32)
        self.end_pos = self._get_random_position().astype(np.float32)

        while (np.linalg.norm(self.start_pos - self.end_pos) < min_distance):
            self.end_pos = self._get_random_position().astype(np.float32)

    def _generate_boundary(self):
        # 简单地设置一个正方形区域为内部，外部为 0，内部为 1
        boundary = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        margin = 10  # 边界宽度
        boundary[margin:-margin, margin:-margin] = 1
        return boundary

    def _generate_elevation(self):
        # 生成随机的高差地图，值域为 0-255
        elevation = np.random.randint(
            low=0, high=256, size=(self.map_size, self.map_size), dtype=np.uint8)
        return elevation

    def _generate_objects(self):
        # 随机放置一些物体
        objects = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        num_trees = 50
        num_structures = 10
        num_landscapes = 5

        for _ in range(num_trees):
            x, y = self._get_random_position()
            objects[y, x] = 1  # 树木标记为 1

        for _ in range(num_structures):
            x, y = self._get_random_position()
            objects[y, x] = 2  # 保留构筑物标记为 2

        for _ in range(num_landscapes):
            x, y = self._get_random_position()
            objects[y, x] = 3  # 人工景观标记为 3

        return objects

    def _get_random_position(self):
        # 在红线内随机获取一个位置，返回整数坐标
        while True:
            x = random.randint(0, self.map_size - 1)
            y = random.randint(0, self.map_size - 1)
            if self.map[0, y, x] == 1:
                return np.array([x, y], dtype=np.int32)

    def reset(self):
        self._generate_map()
        self.agent_positions = [self.start_pos.copy()
                                for _ in range(self.num_agents)]
        self.trajectories = [[] for _ in range(self.num_agents)]
        self.steps = 0
        obs = self._get_observations()
        if self.render_mode:
            self.render()
        return obs

    def _get_observations(self):
        # 获取智能体的观测，返回每个智能体的完整地图
        return [self.map.copy() for _ in range(self.num_agents)]

    def step(self, actions):
        rewards = []
        dones = []
        infos = {}
        self.steps += 1

        for i, action in enumerate(actions):
            old_pos = self.agent_positions[i].copy()
            new_pos = self._move_agent(self.agent_positions[i], action)
            self.agent_positions[i] = new_pos
            self.trajectories[i].append(new_pos.copy())

            # 计算奖励
            reward = self._calculate_reward(
                old_pos, new_pos, self.trajectories[i])
            rewards.append(reward)

            # 判断是否到达终点或达到最大步数
            done = self._is_done(new_pos) or self.steps >= self.max_steps
            dones.append(done)

            # 在通道 2 中标记智能体的轨迹，值为 255
            x, y = int(new_pos[0]), int(new_pos[1])
            self.map[2, y, x] = 255

        obs = self._get_observations()

        if self.render_mode:
            self.render()

        return obs, rewards, dones, infos

    def _move_agent(self, position, action):
        # 动作为二维连续值，表示在 x 和 y 方向的速度，范围在 [-1, 1]
        # 我们将速度映射到实际的移动步长，这里可以乘以一个系数
        speed_scale = 1.0  # 控制移动速度的系数
        delta = action * speed_scale

        # 更新位置
        new_pos = position + delta

        # 确保新位置在地图范围内
        new_pos[0] = np.clip(new_pos[0], 0, self.map_size - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.map_size - 1)

        # 检查是否在红线内
        x, y = int(new_pos[0]), int(new_pos[1])
        if self.map[0, y, x] == 1:
            return new_pos
        else:
            # 如果超出红线，保持原位
            return position

    def _calculate_reward(self, old_pos, new_pos, trajectory):
        reward = 0

        # 将位置转换为整数索引
        x_old, y_old = int(old_pos[0]), int(old_pos[1])
        x_new, y_new = int(new_pos[0]), int(new_pos[1])

        # 高差奖励
        old_elevation = self.map[1, y_old, x_old]
        new_elevation = self.map[1, y_new, x_new]
        if abs(int(new_elevation) - int(old_elevation)) < 5:
            reward += 1  # 高差变化小，加分

        # 绕圈或走回头路惩罚
        if any((np.array_equal(new_pos, pos) for pos in trajectory[:-1])):
            reward -= 2  # 走回头路，扣分

        # 经过兴趣点奖励或惩罚
        obj_value = self.map[2, y_new, x_new]
        if obj_value in [1, 2, 3]:
            reward += 1  # 经过树木等兴趣点，加分
            if obj_value == 1:
                reward -= 1  # 与树木重合，扣分

        return reward

    def _is_done(self, position):
        # 判断是否到达终点附近
        if np.linalg.norm(position - self.end_pos) < 5:
            return True
        else:
            return False

    def render(self, mode='human'):
        # 实时显示地图和智能体位置
        plt.figure(figsize=(6, 6))
        # 绘制地块红线
        boundary = self.map[0] * 255
        plt.imshow(boundary, cmap='gray', alpha=0.3)

        # 绘制高差（仅作为背景，不影响显示效果）
        elevation = self.map[1]
        plt.imshow(elevation, cmap='terrain', alpha=0.3)

        # 绘制物体
        objects = self.map[2]
        plt.imshow(objects, cmap='jet', alpha=0.5)

        # 绘制智能体轨迹和当前位置
        for i, traj in enumerate(self.trajectories):
            traj = np.array(traj)
            if traj.size > 0:
                plt.plot(traj[:, 0], traj[:, 1], marker='o',
                         label=f'Agent {i+1}')
                plt.scatter(self.agent_positions[i][0],
                            self.agent_positions[i][1], s=100)
                
        # 绘制起点和终点
        plt.scatter(self.start_pos[0], self.start_pos[1],
                    marker='*', s=200, c='green', label='Start')
        plt.scatter(self.end_pos[0], self.end_pos[1],
                    marker='X', s=200, c='red', label='End')

        plt.legend()
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        plt.title('Park Environment')
        plt.show()
        plt.close()

    def close(self):
        pass
