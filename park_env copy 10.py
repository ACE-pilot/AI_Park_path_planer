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

    def __init__(self, num_agents=1, render_mode=False):
        super(ParkEnv, self).__init__()
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.map_size = 32
        self.observation_shape = (3, self.map_size, self.map_size)  # 保持为 3 个通道

        self.total_steps = 0  # 新增：记录总步数
        self.total_episodes = 0  # 新增：记录总回合数

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
        
        # 定义起点和终点的对象层标记值
        self.START_POINT_VALUE = 40
        self.END_POINT_VALUE = 50
        
        # 初始化环境
        self._generate_map()
        self.agent_positions = []
        self.trajectories = [[] for _ in range(self.num_agents)]
        self.steps = 0
        self.max_steps = 20  # 最大步数，防止无限循环

        # 奖励和惩罚参数
        self.endpoint_bonus = 500  # 智能体到达终点的奖励分数
        self.endpoint_penalty = 100  # 智能体未到达终点的惩罚分数
        


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

            # 将起点和终点标记到对象层，确保索引在有效范围内
            x_start, y_start = int(round(self.start_pos[0])), int(round(self.start_pos[1]))
            x_end, y_end = int(round(self.end_pos[0])), int(round(self.end_pos[1]))

            # 使用 np.clip 确保索引在 [0, self.map_size - 1] 范围内
            x_start = np.clip(x_start, 0, self.map_size - 1)
            y_start = np.clip(y_start, 0, self.map_size - 1)
            x_end = np.clip(x_end, 0, self.map_size - 1)
            y_end = np.clip(y_end, 0, self.map_size - 1)

            # 定义标记区域的距离
            distance = self.map_size//12

            # 标记起点附近的区域
            y_start_min = max(y_start - distance, 0)
            y_start_max = min(y_start + distance + 1, self.map_size)
            x_start_min = max(x_start - distance, 0)
            x_start_max = min(x_start + distance + 1, self.map_size)
            self.map[2, y_start_min:y_start_max, x_start_min:x_start_max] = self.START_POINT_VALUE  # 标记起点附近区域

            # 标记终点附近的区域
            y_end_min = max(y_end - distance, 0)
            y_end_max = min(y_end + distance + 1, self.map_size)
            x_end_min = max(x_end - distance, 0)
            x_end_max = min(x_end + distance + 1, self.map_size)
            self.map[2, y_end_min:y_end_max, x_end_min:x_end_max] = self.END_POINT_VALUE        # 标记终点附近区域

            logger.debug(f"起点 ({x_start}, {y_start}) 和终点 ({x_end}, {y_end}) 已标记到对象层。")
            
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
        #随机选择边界上的一点（包括顶点和线段上的点）。
        boundary = self.boundary_polygon.exterior
        total_length = boundary.length
        random_distance = random.uniform(0, total_length)
        point = boundary.interpolate(random_distance)
        return np.array([point.x, point.y], dtype=np.float32)


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

        # # 设置树木密度和兴趣点数量，如果未指定则随机生成
        # if self.tree_density is None:
        #     num_trees = random.randint(15, 35)  # 树木数量范围
        # else:
        #     num_trees = int(self.tree_density * self.map_size * self.map_size)

        # if self.num_interest_points is None:
        #     num_structures = random.randint(2, 7)
        #     num_landscapes = random.randint(1, 4)
        # else:
        #     num_structures = self.num_interest_points
        #     num_landscapes = self.num_interest_points // 2

        # # 树木，以圆形表示，可以重叠，形成簇
        # for _ in range(num_trees):
        #     x, y = self._get_random_position()
        #     radius = random.randint(2, 5)
        #     rr, cc = sk_circle((y, x), radius, shape=objects.shape)
        #     objects[rr, cc] = 1  # 树木标记为 1

        # # 保留构筑物，以方形表示
        # for _ in range(num_structures):
        #     x, y = self._get_random_position()
        #     size = random.randint(2, 4)
        #     half_size = size // 2
        #     x_min = max(0, x - half_size)
        #     x_max = min(self.map_size, x + half_size)
        #     y_min = max(0, y - half_size)
        #     y_max = min(self.map_size, y + half_size)
        #     objects[y_min:y_max, x_min:x_max] = 2  # 保留构筑物标记为 2

        # # 人工景观，以较大的圆形表示
        # for _ in range(num_landscapes):
        #     x, y = self._get_random_position()
        #     radius = random.randint(2, 5)
        #     rr, cc = sk_circle((y, x), radius, shape=objects.shape)
        #     objects[rr, cc] = 3  # 人工景观标记为 3

        return objects

    def _get_random_position(self):
        # 在红线内随机获取一个位置，返回整数坐标
        while True:
            x = random.randint(0, self.map_size - 1)
            y = random.randint(0, self.map_size - 1)
            if self.map[0, y, x] == 1:
                return np.array([x, y], dtype=np.int32)

    def reset(self, change_env_every_n_episodes=40000):
        # 如果环境需要在特定回合数后重置随机生成的参数
        if self.total_episodes % change_env_every_n_episodes == 0:
            self._generate_map()  # 随机化地图
            self.agent_positions = [self.start_pos.copy() for _ in range(self.num_agents)]
            self.trajectories = [[] for _ in range(self.num_agents)]
            self._clear_agent_trails() # 在回合开始时清除轨迹
        else:
            # 保持之前的地图，只重置智能体位置和轨迹
            self.agent_positions = [self.start_pos.copy() for _ in range(self.num_agents)]
            self.trajectories = [[] for _ in range(self.num_agents)]
            self._clear_agent_trails() # 在回合开始时清除轨迹
        
        self.steps = 0
        # 初始化每个智能体是否到达终点的标志
        self.agent_reached_goal = [False for _ in range(self.num_agents)]
        obs = self._get_observations()
        if self.render_mode:
            self.render()
        return obs

    #=========================================================================   临时关闭两个通道   ==================================================================    
   
    # def _get_observations(self):
    #     # 获取每个智能体的观测，返回一个字典
    #     obs = {agent: self.map.copy() for agent in self.agents}
    #     return obs
    
    def _get_observations(self):
        # 创建一个观测副本
        observations = self.map.copy()
        
        # 将前两个通道（边界层和高程层）设置为 0
        observations[0] = 0  # 第一个通道（边界层）
        observations[1] = 0  # 第二个通道（高程层）
        
        # 返回包含三个通道的观测，每个智能体只会看到物件层（第三个通道有效）
        obs = {agent: observations for agent in self.agents}
        return obs


    def step(self, actions):
        # actions: 包含每个智能体动作的字典，键为智能体名称
        rewards = {}
        dones = {}
        infos = {}
        self.steps += 1

        # 清除之前的智能体轨迹(每个回合内不清除)
        # self._clear_agent_trails()

        for i, agent in enumerate(self.agents):
            # 如果该智能体已经到达终点，则不再执行动作
            if self.agent_reached_goal[i]:
                rewards[agent] = 0  # 已到达终点的智能体不再获得或扣除奖励
                dones[agent] = True  # 标记该智能体停止
                continue

            # 获取智能体的动作并执行移动
            action = actions[agent]
            old_pos = self.agent_positions[i].copy()
            new_pos = self._move_agent(self.agent_positions[i], action)
            self.agent_positions[i] = new_pos
            self.trajectories[i].append(new_pos.copy())

            # 更新智能体轨迹和位置到通道 2
            self._update_agent_trail(i, new_pos)

            # 计算奖励
            reward = self._calculate_reward(old_pos, new_pos, self.trajectories[i], i)

            # 判断智能体是否到达终点
            if self._is_done(new_pos):
                self.agent_reached_goal[i] = True
                reward += self.endpoint_bonus  # 智能体到达终点附近，加上额外的奖励
                dones[agent] = True  # 该智能体停止
            else:
                dones[agent] = False  # 未到达终点，继续进行

            rewards[agent] = reward
            infos[agent] = {}

        # 检查是否达到最大步数或所有智能体都到达终点
        if self.steps >= self.max_steps or all(self.agent_reached_goal):
            # 回合结束，所有智能体的 done 都为 True
            for i, agent in enumerate(self.agents):
                dones[agent] = True
                # 对于未到达终点的智能体，给予额外的惩罚
                if not self.agent_reached_goal[i]:
                    rewards[agent] -= self.endpoint_penalty  # 智能体未到达终点，扣分

        # 获取新的观测值
        obs = self._get_observations()
        if self.render_mode:
            self.render()
        return obs, rewards, dones, infos


    def _clear_agent_trails(self):
        # 将通道 2 中之前添加的智能体轨迹值清除
        self.map[2][self.map[2] >= 100] = 0


    def _update_agent_trail(self, agent_idx, new_pos):
        x, y = int(round(new_pos[0])), int(round(new_pos[1]))

        # 使用 np.clip 确保索引在 [0, self.map_size - 1] 范围内
        x = np.clip(x, 0, self.map_size - 1)
        y = np.clip(y, 0, self.map_size - 1)

        # 定义轨迹标记值 (100 + agent_index)
        agent_trail_value = 100 + agent_idx*5

        # 更新轨迹: 之前的位置作为轨迹
        if len(self.trajectories[agent_idx]) > 1:
            old_pos = self.trajectories[agent_idx][-1]
            old_x, old_y = int(round(old_pos[0])), int(round(old_pos[1]))
            old_x = np.clip(old_x, 0, self.map_size - 1)
            old_y = np.clip(old_y, 0, self.map_size - 1)
            self.map[2, old_y, old_x] = agent_trail_value  # 标记为轨迹值

        # 定义当前位置标记值为 (200 + agent_index)，每个智能体有不同的当前位置标记
        agent_current_value = 200 + agent_idx*5
        self.map[2, y, x] = agent_current_value

        # 更新智能体的轨迹
        self.trajectories[agent_idx].append(new_pos)
        self.agent_positions[agent_idx] = new_pos



    def _move_agent(self, position, action):
        # 将 action 转换为 NumPy 数组
        action = np.array(action, dtype=np.float32)

        # 动作为二维连续值，表示在 x 和 y 方向的速度，范围在 [-1, 1]
        speed_scale = 4  # 控制移动速度的系数
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
        # 每走一步都扣 0.1 分，增加移动成本
        reward -= 1
        
        # 将位置转换为整数索引
        x_old, y_old = int(round(old_pos[0])), int(round(old_pos[1]))
        x_new, y_new = int(round(new_pos[0])), int(round(new_pos[1]))

        # 使用 np.clip 确保索引在 [0, self.map_size - 1] 范围内
        x_old = np.clip(x_old, 0, self.map_size - 1)
        y_old = np.clip(y_old, 0, self.map_size - 1)
        x_new = np.clip(x_new, 0, self.map_size - 1)
        y_new = np.clip(y_new, 0, self.map_size - 1)

        # 计算到终点的距离变化，鼓励向目标移动
        old_distance = np.linalg.norm(old_pos - self.end_pos)
        new_distance = np.linalg.norm(new_pos - self.end_pos)
        distance_delta = old_distance - new_distance

        if distance_delta > 0:
            reward += 0.2 * distance_delta  # 向终点移动的奖励
        else:
            reward -= 0.1 * abs(distance_delta)  # 远离终点的惩罚

        # # 高差奖励
        # old_elevation = self.map[1, y_old, x_old]
        # new_elevation = self.map[1, y_new, x_new]
        # if abs(int(new_elevation) - int(old_elevation)) < 5:
        #     reward += 1  # 高差变化小，加分

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
                reward -= 0.2  # 绕圈，扣分

        # # 获取兴趣点的坐标列表（不包括树木的中心点）
        # interest_points = self._get_interest_points(exclude_tree_centers=True)

        # # 计算到兴趣点的最小距离
        # min_distance = min(
        #     np.linalg.norm(new_pos - point) for point in interest_points
        # ) if interest_points else float('inf')

        # # 距离奖励函数，距离越近奖励越高，超过 20 像素则不奖励
        # max_reward_distance = 20.0
        # n = 2  # 可以根据需要调整 n 的值
        # if min_distance <= max_reward_distance:
        #     proximity_reward = ((max_reward_distance - min_distance) / max_reward_distance) ** n
        #     reward += proximity_reward

        # # 检查是否与树木中心点或其他兴趣点的位置重合（碰撞）
        # collision_points = self._get_interest_points(include_tree_centers=True)
        # if any(np.array_equal(new_pos.astype(int), point.astype(int)) for point in collision_points):
        #     reward -= 1  # 与树木中心点或兴趣点重合，扣分

        return reward


    def _is_done(self, position):
        # 检查智能体是否到达共享的终点
        if np.linalg.norm(position - self.end_pos) < 5:
            return True
        else:
            return False



    def render(self, mode='human'):
        # 创建一个包含主图和三个附图的图形，布局为1行4列
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle('Park Environment Visualization', fontsize=16)

        # ------------------ 主图：综合环境视图 ------------------
        ax_main = axs[0]
        ax_main.set_title('Main Environment View')

        # 绘制地块红线
        boundary = self.map[0] * 255
        ax_main.imshow(boundary, cmap='gray', alpha=0.3)

        # 绘制高程
        elevation = self.map[1]
        im_elevation = ax_main.imshow(elevation, cmap='terrain', alpha=0.3)

        # 绘制等高线
        contour_levels = np.linspace(0, 255, 10)
        contours = ax_main.contour(elevation, levels=contour_levels, cmap='terrain', alpha=0.5)
        ax_main.clabel(contours, inline=True, fontsize=8)

        # 绘制物体
        objects = self.map[2]

        # 绘制物体图层
        im_objects = ax_main.imshow(objects, cmap='jet', alpha=0.5)

        # 绘制智能体轨迹和当前位置
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'red']
        for i, agent in enumerate(self.agents):
            traj = np.array(self.trajectories[i])
            if traj.size > 0:
                ax_main.plot(traj[:, 0], traj[:, 1], marker='o', label=f'Agent {i+1} Trajectory', color=colors[i % len(colors)])
                ax_main.scatter(self.agent_positions[i][0], self.agent_positions[i][1], s=100, color=colors[i % len(colors)], edgecolors='white', label=f'Agent {i+1} Position')

        # 绘制共享的起点和终点
        ax_main.scatter(self.start_pos[0], self.start_pos[1], marker='*', s=200, c='green', edgecolors='black', label='Start')
        ax_main.scatter(self.end_pos[0], self.end_pos[1], marker='X', s=200, c='red', edgecolors='black', label='End')

        # 创建主图的图例，避免重复标签
        handles, labels = ax_main.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_main.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

        ax_main.set_xlim(0, self.map_size)
        ax_main.set_ylim(self.map_size, 0)  # y 轴反转以匹配图像坐标系

        # ------------------ 附图 1：边界层 ------------------
        ax_boundary = axs[1]
        ax_boundary.set_title('Boundary Layer')
        ax_boundary.imshow(self.map[0], cmap='gray')
        ax_boundary.axis('off')

        # 创建图例
        ax_boundary.scatter([], [], c='gray', marker='s', label='Boundary')
        ax_boundary.legend(loc='upper right', fontsize='small')

        # ------------------ 附图 2：高程层 ------------------
        ax_elevation = axs[2]
        ax_elevation.set_title('Elevation Layer')
        im_elevation_layer = ax_elevation.imshow(self.map[1], cmap='terrain')
        
        # 绘制等高线
        contours_elevation = ax_elevation.contour(self.map[1], levels=contour_levels, cmap='terrain', alpha=0.5)
        ax_elevation.clabel(contours_elevation, inline=True, fontsize=8)
        
        ax_elevation.axis('off')

        # 添加颜色条
        cbar = fig.colorbar(im_elevation_layer, ax=ax_elevation, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Elevation')

        # ------------------ 附图 3：物体层 ------------------
        ax_objects = axs[3]
        ax_objects.set_title('Objects Layer')

        # 绘制物体层，使用更合理的颜色
        objects = self.map[2]
        trees_mask = objects == 1
        structures_mask = objects == 2
        landscapes_mask = objects == 3
        start_mask = objects == self.START_POINT_VALUE
        end_mask = objects == self.END_POINT_VALUE

        # 绘制树木
        ax_objects.scatter(
            np.where(trees_mask)[1], np.where(trees_mask)[0],
            c='green', marker='^', s=50, label='Trees'
        )

        # 绘制构筑物
        ax_objects.scatter(
            np.where(structures_mask)[1], np.where(structures_mask)[0],
            c='gray', marker='s', s=50, label='Structures'
        )

        # 绘制人工景观
        ax_objects.scatter(
            np.where(landscapes_mask)[1], np.where(landscapes_mask)[0],
            c='blue', marker='o', s=100, label='Artificial Landscapes'
        )

        # 绘制起点和终点
        ax_objects.scatter(
            self.start_pos[0], self.start_pos[1],
            marker='*', s=200, c='yellow', edgecolors='black', label='Start'
        )
        ax_objects.scatter(
            self.end_pos[0], self.end_pos[1],
            marker='X', s=200, c='red', edgecolors='black', label='End'
        )

        # 反转 Y 轴以匹配图像坐标系
        ax_objects.set_ylim(self.map_size, 0)  # 将 Y 轴方向反转

        # 创建图例
        ax_objects.scatter([], [], c='green', marker='^', label='Trees')
        ax_objects.scatter([], [], c='gray', marker='s', label='Structures')
        ax_objects.scatter([], [], c='blue', marker='o', label='Artificial Landscapes')
        ax_objects.scatter([], [], c='yellow', marker='*', edgecolors='black', label='Start')
        ax_objects.scatter([], [], c='red', marker='X', edgecolors='black', label='End')
        ax_objects.legend(loc='upper right', fontsize='small')

        ax_objects.axis('off')

        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=True)
        plt.pause(0.001)  # 非阻塞暂停，适合动画渲染
        plt.close(fig)




    def close(self):
        pass
