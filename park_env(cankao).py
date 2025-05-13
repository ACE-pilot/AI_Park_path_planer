import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import random
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import scale, translate
from skimage.draw import disk as sk_circle
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s %(threadName)s @%(filename)s:%(lineno)d] %(message)s')
logger = logging.getLogger(__name__)

class ParkEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents=3, render_mode=False):
        super(ParkEnv, self).__init__()
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.map_size = 64  # 调整地图尺寸
        self.observation_shape = (3, self.map_size, self.map_size)  # 保持为 3 个通道

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
        logger.debug("开始生成地图。")
        try:
            # 生成地图
            self.map = np.zeros(self.observation_shape, dtype=np.uint8)
            # 通道 0：地块红线
            self.map[0], self.boundary_polygon = self._generate_boundary()
            logger.debug("边界生成完成。")

            # 通道 1：高程
            self.map[1] = self._generate_elevation()
            logger.debug("高程生成完成。")

            # 通道 2：物体
            self.map[2] = self._generate_objects()
            logger.debug("物体生成完成。")

            # 初始化智能体轨迹数组
            self.agent_trail = np.zeros((self.map_size, self.map_size), dtype=np.uint8)

            # 设置起点和终点（所有智能体共享）
            self.start_pos, self.end_pos = self._select_start_end_points()
            logger.debug("起点和终点设置完成。")

        except ValueError as e:
            logger.error(f"地图生成失败：{e}")
            raise e

    def _generate_boundary(self):
        logger.debug("开始生成边界。")
        max_attempts = 20
        attempts = 0
        min_area_threshold = (self.map_size ** 2) * 0.2  # 设置最小面积阈值，根据需要调整

        while attempts < max_attempts:
            # 使用规则性分布生成点集
            num_points = random.randint(3, 8)  # 保持点的数量为3到8
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)  # 均匀分布角度
            radius = self.map_size / 2 - 2  # 留出边距
            points = np.stack((radius * np.cos(angles), radius * np.sin(angles)), axis=1)
            # 添加少量随机噪声
            noise = np.random.uniform(-self.map_size * 0.05, self.map_size * 0.05, size=points.shape)
            points += noise
            points = points + self.map_size / 2  # 平移到地图中心

            # 确保所有点在地图内
            points = np.clip(points, 2, self.map_size - 2).astype(int)

            # 检查是否有足够的唯一点
            if len(np.unique(points, axis=0)) < 3:
                logger.debug("生成的点集不足以构建凸包，重新生成点集。")
                attempts += 1
                continue

            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                polygon = Polygon(hull_points)

                if not polygon.is_valid or polygon.is_empty:
                    logger.debug("生成的多边形无效或为空，重新生成点集。")
                    attempts += 1
                    continue

                # 计算凸包面积
                area = polygon.area
                logger.debug(f"生成的凸包面积: {area}")
                if area < min_area_threshold:
                    logger.debug("凸包面积过小，进行点集调整。")
                    # 使用PCA找出主方向
                    pca = PCA(n_components=2)
                    pca.fit(points)
                    components = pca.components_
                    # 计算垂直方向
                    perpendicular = np.array([-components[0,1], components[0,0]])
                    # 偏移每个点
                    offset_distance = self.map_size * 0.05  # 根据需要调整偏移距离
                    points = points + perpendicular * offset_distance
                    points = np.clip(points, 2, self.map_size - 2).astype(int)
                    # 重新计算凸包
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    polygon = Polygon(hull_points)
                    area = polygon.area
                    logger.debug(f"调整后的凸包面积: {area}")
                    if area < min_area_threshold:
                        logger.debug("调整后凸包面积仍然过小，重新生成点集。")
                        attempts += 1
                        continue

                # 计算缩放比例，使多边形与外部矩形相接
                centroid = polygon.centroid
                minx, miny, maxx, maxy = polygon.bounds
                scale_x = (self.map_size / (maxx - minx)) * 0.9
                scale_y = (self.map_size / (maxy - miny)) * 0.9
                scale_factor = min(scale_x, scale_y)

                # 缩放多边形
                scaled_polygon = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin=centroid)

                # 平移多边形，使其位于地图中心
                dx = (self.map_size / 2) - scaled_polygon.centroid.x
                dy = (self.map_size / 2) - scaled_polygon.centroid.y
                translated_polygon = translate(scaled_polygon, xoff=dx, yoff=dy)

                # 生成多边形掩码
                y_indices, x_indices = np.indices((self.map_size, self.map_size))
                coords = np.stack((x_indices.ravel(), y_indices.ravel()), axis=-1)
                mask = np.array([translated_polygon.contains(Point(x, y)) for x, y in coords]).reshape(self.map_size, self.map_size)
                boundary = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
                boundary[mask] = 1

                # 正确标记边界线
                boundary_line = translated_polygon.boundary
                boundary_coords = list(boundary_line.coords)
                for point in boundary_coords:
                    x, y = int(round(point[0])), int(round(point[1]))
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        boundary[y, x] = 1  # 标记边界线上的点为1

                if not boundary.any():
                    logger.debug("生成的边界掩码中没有任何有效点，重新生成点集。")
                    attempts += 1
                    continue

                logger.debug("边界掩码生成成功。")
                return boundary, translated_polygon

            except ConvexHull.QhullError as e:
                logger.error(f"ConvexHull 计算失败：{e}，重新生成点集。")
                attempts += 1
                continue

    def _get_random_point_on_boundary(self):
        """随机选择边界上的一点（包括顶点和线段上的点）。"""
        boundary = self.boundary_polygon.exterior
        total_length = boundary.length
        random_distance = random.uniform(0, total_length)
        point = boundary.interpolate(random_distance)
        return np.array([point.x, point.y], dtype=np.float32)

    def _get_random_inside_position(self):
        """在边界内随机选择一个点。"""
        while True:
            x = random.uniform(0, self.map_size)
            y = random.uniform(0, self.map_size)
            point = Point(x, y)
            if self.boundary_polygon.contains(point):
                return np.array([x, y], dtype=np.float32)

    def _select_start_end_points(self):
        logger.debug("开始选择起点和终点。")
        min_distance = self.map_size * 0.3  # 设定最小距离
        max_attempts = 10
        buffer_distance = self.map_size * 0.05  # 设定缓冲距离，沿边界线移动

        # 随机选择起点
        start_pos = self._get_random_point_on_boundary()
        logger.debug(f"选择的起点: {start_pos}")

        # 尝试找到满足最小距离的终点
        for attempt in range(1, max_attempts + 1):
            end_pos = self._get_random_point_on_boundary()
            distance = np.linalg.norm(start_pos - end_pos)
            logger.debug(f"尝试 {attempt}: 终点 {end_pos} 距离起点 {distance}")
            if distance >= min_distance:
                logger.debug(f"找到满足最小距离的终点，距离: {distance}")
                return start_pos, end_pos

        # 如果尝试失败，选择距离最远的点
        logger.debug("尝试找到满足最小距离的终点失败，选择最远点。")
        boundary = self.boundary_polygon.exterior
        points = np.array(boundary.coords[:-1])  # 去掉重复的最后一个点
        distances = np.linalg.norm(points - start_pos, axis=1)
        max_distance_idx = np.argmax(distances)
        farthest_point = points[max_distance_idx]
        logger.debug(f"选择的最远点（未经偏移）: {farthest_point}")

        # 添加少量随机沿边界线的偏移，确保终点仍在边界线上
        boundary_line = self.boundary_polygon.exterior
        total_length = boundary_line.length
        farthest_point_shapely = Point(farthest_point)
        distance_along_boundary = boundary_line.project(farthest_point_shapely)
        
        # 随机选择移动方向：正向或反向
        direction = random.choice([-1, 1])
        shift_distance = random.uniform(0, buffer_distance) * direction
        new_distance = (distance_along_boundary + shift_distance) % total_length
        new_end_pos_shapely = boundary_line.interpolate(new_distance)
        new_end_pos = np.array([new_end_pos_shapely.x, new_end_pos_shapely.y], dtype=np.float32)
        distance = np.linalg.norm(start_pos - new_end_pos)
        logger.debug(f"选择的终点（经过沿边界线偏移）: {new_end_pos} 距离: {distance}")

        if distance >= min_distance:
            logger.debug(f"选择的终点满足最小距离要求，距离: {distance}")
            return start_pos, new_end_pos
        else:
            logger.error("无法生成足够远的起点和终点。")
            raise ValueError("无法生成足够远的起点和终点。")

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
            num_trees = random.randint(5, 30)  # 树木数量范围
        else:
            num_trees = int(self.tree_density * self.map_size * self.map_size)

        if self.num_interest_points is None:
            num_structures = random.randint(1, 5)
            num_landscapes = random.randint(1, 3)
        else:
            num_structures = self.num_interest_points
            num_landscapes = self.num_interest_points // 2

        # 树木，以圆形表示，可以重叠，形成簇
        for _ in range(num_trees):
            try:
                x, y = self._get_random_inside_position()
                radius = random.randint(2, 6)
                rr, cc = sk_circle(int(y), int(x), radius, shape=objects.shape)
                objects[rr, cc] = 1  # 树木标记为 1
            except Exception as e:
                logger.warning(f"生成树木时出错：{e}")

        # 保留构筑物，以方形表示
        for _ in range(num_structures):
            try:
                x, y = self._get_random_inside_position()
                size = random.randint(1, 4)
                half_size = size // 2
                x_min = max(0, int(x) - half_size)
                x_max = min(self.map_size, int(x) + half_size + 1)
                y_min = max(0, int(y) - half_size)
                y_max = min(self.map_size, int(y) + half_size + 1)
                objects[y_min:y_max, x_min:x_max] = 2  # 保留构筑物标记为 2
            except Exception as e:
                logger.warning(f"生成构筑物时出错：{e}")

        # 人工景观，以较大的圆形表示
        for _ in range(num_landscapes):
            try:
                x, y = self._get_random_inside_position()
                radius = random.randint(2, 6)
                rr, cc = sk_circle(int(y), int(x), radius, shape=objects.shape)
                objects[rr, cc] = 3  # 人工景观标记为 3
            except Exception as e:
                logger.warning(f"生成人工景观时出错：{e}")

        return objects

    def reset(self):
        try:
            self._generate_map()
        except ValueError as e:
            logger.error(f"地图生成失败：{e}")
            raise e

        self.agent_positions = [self.start_pos.copy() for _ in range(self.num_agents)]
        self.trajectories = [[] for _ in range(self.num_agents)]
        self.steps = 0
        # 初始化每个智能体是否到达终点的标志
        self.agent_reached_goal = [False for _ in range(self.num_agents)]
        self._clear_agent_trails()  # 在回合开始时清除轨迹
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

            # 更新智能体轨迹和位置到通道 2
            self._update_agent_trail(i, new_pos)

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

    def _clear_agent_trails(self):
        # 将通道 2 中之前添加的智能体轨迹值清除
        self.map[2][self.map[2] >= 10] = 0
        logger.debug("清除智能体轨迹完成。")

    def _update_agent_trail(self, agent_index, position):
        x, y = int(position[0]), int(position[1])
        agent_value = 10 + agent_index  # 确保与物体的值（1、2、3）不冲突
        self.map[2, y, x] = agent_value
        logger.debug(f"更新智能体 {agent_index} 的轨迹位置：({x}, {y})")

    def _move_agent(self, position, action):
        # 将 action 转换为 NumPy 数组
        action = np.array(action, dtype=np.float32)

        # 动作为二维连续值，表示在 x 和 y 方向的速度，范围在 [-1, 1]
        speed_scale = 5.0  # 控制移动速度的系数，适当减小以适应更大地图
        delta = action * speed_scale

        # 更新位置
        new_pos = position + delta

        # 确保新位置在地图范围内
        new_pos[0] = np.clip(new_pos[0], 0, self.map_size - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.map_size - 1)

        # 检查是否在边界线内
        point = Point(new_pos[0], new_pos[1])
        if self.boundary_polygon.contains(point):
            logger.debug(f"智能体移动到新位置：({new_pos[0]}, {new_pos[1]})")
            return new_pos
        else:
            logger.debug(f"智能体尝试移动到无效位置：({new_pos[0]}, {new_pos[1]})，保持原位。")
            return position

    def _get_interest_points(self, exclude_tree_centers=False, include_tree_centers=False):
        interest_points = []

        # 获取保留构筑物（值为 2）和人工景观（值为 3）的坐标
        positions = np.argwhere((self.map[2] == 2) | (self.map[2] == 3))
        for pos in positions:
            y, x = pos  # 注意，np.argwhere 返回 (行, 列)
            interest_points.append(np.array([x, y], dtype=np.float32))

        if include_tree_centers:
            # 获取树木中心点（值为 1）的坐标
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

        # 距离奖励函数，距离越近奖励越高，超过 10 像素则不奖励
        max_reward_distance = 10.0 
        n = 2  # 可以根据需要调整 n 的值
        if min_distance <= max_reward_distance:
            proximity_reward = ((max_reward_distance - min_distance) / max_reward_distance) ** n
            reward += proximity_reward

        # 检查是否与树木中心点或其他兴趣点的位置重合（碰撞）
        collision_points = self._get_interest_points(include_tree_centers=True)
        if any(np.array_equal(new_pos.astype(int), point.astype(int)) for point in collision_points):
            reward -= 1  # 与树木中心点或兴趣点重合，扣分

        logger.debug(f"智能体 {agent_index} 奖励计算完成：{reward}")
        return reward

    def _is_done(self, position):
        # 检查智能体是否到达共享的终点
        if np.linalg.norm(position - self.end_pos) < 5:
            logger.debug(f"智能体到达终点：{position}")
            return True
        else:
            return False

    def render(self, mode='human'):
        # 实时显示地图和智能体位置
        plt.figure(figsize=(6, 6))
        plt.axis('off')

        # 绘制高程
        elevation = self.map[1]
        plt.imshow(elevation, cmap='terrain', alpha=0.5)

        # 绘制地块红线
        boundary = self.map[0] * 255
        plt.imshow(boundary, cmap='gray', alpha=0.3)

        # 绘制物体和智能体轨迹
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

        # 绘制物体的实际位置以辅助调试
        trees = np.argwhere(self.map[2] == 1)
        structures = np.argwhere(self.map[2] == 2)
        landscapes = np.argwhere(self.map[2] == 3)
        plt.scatter(trees[:,1], trees[:,0], c='brown', marker='^', label='Trees')
        plt.scatter(structures[:,1], structures[:,0], c='gray', marker='s', label='Structures')
        plt.scatter(landscapes[:,1], landscapes[:,0], c='pink', marker='o', label='Landscapes')

        plt.legend(loc='upper right')
        plt.title('Park Environment')
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        plt.gca().invert_yaxis()  # 以确保y轴方向与图像坐标一致
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    def close(self):
        pass

