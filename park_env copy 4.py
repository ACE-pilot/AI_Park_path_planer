import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import random
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import scale, translate
from skimage.draw import disk as sk_circle
from scipy.spatial import ConvexHull

class ParkEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents=3, render_mode=False):
        super(ParkEnv, self).__init__()
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.map_size = 128
        self.observation_shape = (3, self.map_size, self.map_size)

        # 定义智能体名称列表
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # 动作空间：每个智能体都有一个二维连续动作空间
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_spaces = {agent: self.single_action_space for agent in self.agents}

        # 观察空间：每个智能体都有相同的观察空间
        self.single_observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)
        self.observation_spaces = {agent: self.single_observation_space for agent in self.agents}

        # 环境参数，允许在训练中随机化
        self.tree_density = None  # 树木密度，None 表示随机
        self.num_interest_points = None  # 兴趣点数量，None 表示随机

        # 初始化环境
        self._generate_map()
        self.agent_positions = []
        self.trajectories = [[] for _ in range(self.num_agents)]
        self.steps = 0
        self.max_steps = 50  # 最大步数，防止无限循环
        
        # 奖励和惩罚参数
        self.endpoint_bonus = 10  # 智能体到达终点的奖励分数
        self.endpoint_penalty = 5  # 智能体未到达终点的惩罚分数

    def _generate_map(self):
        # 生成地图
        self.map = np.zeros(self.observation_shape, dtype=np.uint8)
        # 通道 0：地块红线
        self.map[0], self.boundary_polygon = self._generate_boundary()
        # 通道 1：高程
        self.map[1] = self._generate_elevation()
        # 通道 2：物体
        self.map[2] = self._generate_objects()

        # 初始化智能体轨迹数组
        self.agent_trail = np.zeros((self.map_size, self.map_size), dtype=np.uint8)

        # 设置起点和终点（所有智能体共享）
        min_distance = self.map_size * 0.5  # 确保起点和终点有足够的距离
        self.start_pos = self._get_point_on_boundary().astype(np.float32)
        self.end_pos = self._get_point_on_boundary().astype(np.float32)

        while np.linalg.norm(self.start_pos - self.end_pos) < min_distance:
            self.end_pos = self._get_point_on_boundary().astype(np.float32)

    def _generate_boundary(self):
        # 生成随机多边形作为地块红线
        num_points = random.randint(4, 8)
        margin = 10  # 保证点在地图内部
        # 随机在框内部撒点
        points = np.random.randint(margin, self.map_size - margin, size=(num_points, 2))
        # 计算凸包，连接最外圈的点
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # 创建多边形
        polygon = Polygon(hull_points)
        # 计算多边形的中心
        centroid = polygon.centroid

        # 计算缩放比例，使多边形与外部矩形相接
        minx, miny, maxx, maxy = polygon.bounds
        scale_x = (self.map_size / (maxx - minx)) * 0.9  # 留一点边距
        scale_y = (self.map_size / (maxy - miny)) * 0.9
        scale_factor = min(scale_x, scale_y)

        # 以多边形的中心进行缩放
        scaled_polygon = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin=centroid)

        # 平移多边形，使其位于地图中心
        dx = (self.map_size / 2) - scaled_polygon.centroid.x
        dy = (self.map_size / 2) - scaled_polygon.centroid.y
        translated_polygon = translate(scaled_polygon, xoff=dx, yoff=dy)

        # 使用 shapely 生成多边形掩码
        y_indices, x_indices = np.indices((self.map_size, self.map_size))
        coords = np.stack((x_indices.ravel(), y_indices.ravel()), axis=-1)
        mask = np.array([translated_polygon.contains(Point(x, y)) for x, y in coords]).reshape(self.map_size, self.map_size)
        boundary = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        boundary[mask] = 1
        return boundary, translated_polygon

    def _get_point_on_boundary(self):
        # 在边界的边上随机获取一个点
        exterior_coords = list(self.boundary_polygon.exterior.coords)
        edge_indices = list(range(len(exterior_coords) - 1))
        random_edge_index = random.choice(edge_indices)
        point1 = exterior_coords[random_edge_index]
        point2 = exterior_coords[random_edge_index + 1]
        t = random.uniform(0, 1)
        x = point1[0] + t * (point2[0] - point1[0])
        y = point1[1] + t * (point2[1] - point1[1])
        x = np.clip(x, 0, self.map_size - 1)
        y = np.clip(y, 0, self.map_size - 1)
        return np.array([x, y], dtype=np.float32)

    def _generate_elevation(self):
        # 生成具有主要高程点的渐变地形
        elevation = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        num_peaks = random.randint(0, 5)
        peak_positions = np.random.randint(0, self.map_size, size=(num_peaks, 2))
        peak_heights = np.random.uniform(100, 255, size=num_peaks)

        for pos, height in zip(peak_positions, peak_heights):
            x0, y0 = pos
            xx, yy = np.meshgrid(np.arange(self.map_size), np.arange(self.map_size))
            distance = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
            sigma = self.map_size / 8  # 控制山峰的宽度
            elevation += height * np.exp(-distance ** 2 / (2 * sigma ** 2))

        # 归一化到 0-255
        if elevation.max() > elevation.min():
            elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min()) * 255
        else:
            elevation = np.zeros_like(elevation)
        elevation = elevation.astype(np.uint8)
        return elevation

    def _generate_objects(self):
        # 生成具有一定形状和大小的兴趣点
        objects = np.zeros((self.map_size, self.map_size), dtype=np.uint8)

        # 设置树木密度和兴趣点数量，如果未指定则随机生成
        if self.tree_density is None:
            num_trees = random.randint(30, 70)  # 树木数量范围
        else:
            num_trees = int(self.tree_density * self.map_size * self.map_size)

        if self.num_interest_points is None:
            num_structures = random.randint(5, 15)
            num_landscapes = random.randint(3, 8)
        else:
            num_structures = self.num_interest_points
            num_landscapes = self.num_interest_points // 2

        # 树木，以圆形表示，可以重叠，形成簇
        for _ in range(num_trees):
            x, y = self._get_random_position()
            radius = random.randint(2, 5)
            rr, cc = sk_circle((y, x), radius, shape=objects.shape)
            objects[rr, cc] = 1  # 树木标记为 1

        # 保留构筑物，以方形表示
        for _ in range(num_structures):
            x, y = self._get_random_position()
            size = random.randint(4, 8)
            half_size = size // 2
            x_min = max(0, x - half_size)
            x_max = min(self.map_size, x + half_size)
            y_min = max(0, y - half_size)
            y_max = min(self.map_size, y + half_size)
            objects[y_min:y_max, x_min:x_max] = 2  # 保留构筑物标记为 2

        # 人工景观，以较大的圆形表示
        for _ in range(num_landscapes):
            x, y = self._get_random_position()
            radius = random.randint(5, 10)
            rr, cc = sk_circle((y, x), radius, shape=objects.shape)
            objects[rr, cc] = 3  # 人工景观标记为 3

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
        self.agent_positions = [self.start_pos.copy() for _ in range(self.num_agents)]
        self.trajectories = [[] for _ in range(self.num_agents)]
        self.steps = 0
        self.agent_trail = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        # 初始化每个智能体是否到达终点的标志
        self.agent_reached_goal = [False for _ in range(self.num_agents)]
        obs = self._get_observations()
        if self.render_mode:
            self.render()
        return obs


    def _get_observations(self):
        # 获取每个智能体的观测，返回一个字典
        obs = {agent: self.map.copy() for agent in self.agents}
        return obs

    def step(self, actions):
        # actions: 包含每个智能体动作的字典，键为智能体名称
        rewards = {}
        dones = {}
        infos = {}
        self.steps += 1

        for i, agent in enumerate(self.agents):
            action = actions[agent]
            old_pos = self.agent_positions[i].copy()
            new_pos = self._move_agent(self.agent_positions[i], action)
            self.agent_positions[i] = new_pos
            self.trajectories[i].append(new_pos.copy())

            # 计算奖励
            reward = self._calculate_reward(old_pos, new_pos, self.trajectories[i], i)
            
            # 判断智能体是否到达共享的终点
            if self._is_done(new_pos):
                done = True
                self.agent_reached_goal[i] = True
                reward += self.endpoint_bonus  # 智能体到达终点，加分
            else:
                done = False

            rewards[agent] = reward
            dones[agent] = done
            infos[agent] = {}

        # 检查是否达到最大步数，结束回合
        if self.steps >= self.max_steps:
            for i, agent in enumerate(self.agents):
                dones[agent] = True  # 回合结束，所有智能体的 done 都为 True
                if not self.agent_reached_goal[i]:
                    rewards[agent] -= self.endpoint_penalty  # 智能体未到达终点，扣分

        obs = self._get_observations()
        if self.render_mode:
            self.render()
        return obs, rewards, dones, infos


    def _move_agent(self, position, action):
        # 将 action 转换为 NumPy 数组
        action = np.array(action, dtype=np.float32)

        # 动作为二维连续值，表示在 x 和 y 方向的速度，范围在 [-1, 1]
        speed_scale = 10.0  # 控制移动速度的系数
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

    def _get_interest_points(self, exclude_tree_centers=False, include_tree_centers=False):
        interest_points = []

        # 获取保留构筑物（值为2）和人工景观（值为3）的坐标
        positions = np.argwhere((self.map[2] == 2) | (self.map[2] == 3))
        for pos in positions:
            y, x = pos  # 注意，np.argwhere 返回 (行, 列)
            interest_points.append(np.array([x, y], dtype=np.float32))

        if include_tree_centers:
            # 获取树木中心点（值为1）的坐标
            tree_positions = np.argwhere(self.map[2] == 1)
            for pos in tree_positions:
                y, x = pos
                interest_points.append(np.array([x, y], dtype=np.float32))

        return interest_points

    def _calculate_reward(self, old_pos, new_pos, trajectory, agent_index):
        reward = 0

        # 将位置转换为整数索引
        x_old, y_old = int(old_pos[0]), int(old_pos[1])
        x_new, y_new = int(new_pos[0]), int(new_pos[1])

        # 高差奖励
        old_elevation = self.map[1, y_old, x_old]
        new_elevation = self.map[1, y_new, x_new]
        if abs(int(new_elevation) - int(old_elevation)) < 5:
            reward += 1  # 高差变化小，加分

        # 绕圈或走回头路惩罚，使用线段交叉检测
        if len(trajectory) >= 4:
            # 创建当前移动的线段
            current_line = LineString([old_pos, new_pos])
            # 创建之前的轨迹线段集合
            previous_lines = [
                LineString([trajectory[i], trajectory[i + 1]])
                for i in range(len(trajectory) - 2)
            ]
            # 检查当前线段是否与任何之前的线段相交
            if any(current_line.crosses(line) for line in previous_lines):
                reward -= 2  # 绕圈，扣分

        # 获取兴趣点的坐标列表（不包括树木的中心点）
        interest_points = self._get_interest_points(exclude_tree_centers=True)

        # 计算到兴趣点的最小距离
        min_distance = min(
            np.linalg.norm(new_pos - point) for point in interest_points
        ) if interest_points else float('inf')

        # 距离奖励函数，距离越近奖励越高，超过20像素则不奖励
        max_reward_distance = 20.0
        n = 2  # 可以根据需要调整 n 的值
        if min_distance <= max_reward_distance:
            proximity_reward = ((max_reward_distance - min_distance) / max_reward_distance) ** n
            reward += proximity_reward

        # 检查是否与树木中心点或其他兴趣点的位置重合（碰撞）
        collision_points = self._get_interest_points(include_tree_centers=True)
        if any(np.array_equal(new_pos.astype(int), point.astype(int)) for point in collision_points):
            reward -= 1  # 与树木中心点或兴趣点重合，扣分

        return reward


    def _is_done(self, position):
        # 检查智能体是否到达共享的终点
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

        # 绘制等高线
        elevation = self.map[1]
        contour_levels = np.linspace(0, 255, 10)
        plt.contour(elevation, levels=contour_levels, cmap='terrain', alpha=0.5)

        # 绘制高程
        plt.imshow(elevation, cmap='terrain', alpha=0.3)

        # 绘制物体
        objects = self.map[2]
        plt.imshow(objects, cmap='jet', alpha=0.5)

        # 绘制智能体轨迹和当前位置
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'red']
        for i, agent in enumerate(self.agents):
            traj = np.array(self.trajectories[i])
            if traj.size > 0:
                plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f'Agent {i+1}', color=colors[i % len(colors)])
                plt.scatter(self.agent_positions[i][0], self.agent_positions[i][1], s=100, color=colors[i % len(colors)])

        # 绘制共享的起点和终点
        plt.scatter(self.start_pos[0], self.start_pos[1], marker='*', s=200, c='green', label='Start')
        plt.scatter(self.end_pos[0], self.end_pos[1], marker='X', s=200, c='red', label='End')

        plt.legend()
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        plt.title('Park Environment')
        plt.show(block=False) #如果block=True，那么程序会一直等在plt.show那行上
        plt.pause(3)
        plt.close()

    def close(self):
        pass
