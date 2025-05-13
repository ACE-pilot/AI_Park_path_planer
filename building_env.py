import gym
from gym import spaces
import pandas as pd
import socket
import threading
import numpy as np
import random
import json


class GrasshopperEnv(gym.Env):
    def __init__(self, num_agents=9, max_steps=20):
        super(GrasshopperEnv, self).__init__()
        self.num_agents = num_agents
        self.n = self.num_agents
        # 定义动作空间为平面四个方向受力：每个智能体都有4个动作维度，范围从0到1
        # 定义单个智能体的动作空间
        single_agent_action_space = spaces.Box(low=np.array([0]*4), 
                                                   high=np.array([1]*4), dtype=np.float32)
        self.max_steps = max_steps
        self.episode_steps = 0        
        # 为多个智能体定义动作空间列表
        self.action_space = [single_agent_action_space]*self.num_agents
        self.action_spaces = {f"agent_{i}": space for i, space in enumerate(self.action_space)}
        # 定义单个智能体的观察空间
        # 在环境中，每个智能体的观测空间维度是根据几个关键因素计算得出的，具体包括智能体自身的坐标、智能体的四角坐标、智能体相对于其他智能体四角的位置，以及智能体相对于不可动物体的位置。具体来说：
        # 1. **智能体坐标**：每个智能体的坐标由一个二维向量（x, y）表示，因此这部分贡献了2个维度。
        # 2. **自身四角坐标**：每个智能体有四个角点，每个角点由一个二维向量（x, y）表示，因此这部分贡献了\(4 \times 2 = 8\)个维度。
        # 3. **相对于其他智能体四角的位置**：每个智能体需要计算其每个角点相对于其他所有智能体的四个角点的位置。假设有9个智能体，则每个智能体会与其他8个智能体进行比较。每次比较涉及到4个自身角点与另一个智能体的4个角点，每对角点的相对位置由一个二维向量表示，因此这部分贡献了\(8 \times 4 \times 4 \times 2 = 256\)个维度。
        # 4. **相对于不可动物体的位置**：每个智能体的每个角点还需要计算相对于15个不可动物体的位置。每个相对位置由一个二维向量表示，因此这部分贡献了\(15 \times 4 \times 2 = 120\)个维度。
        # 综合以上各部分，每个智能体的观测空间维度总和为\(2 + 8 + 256 + 120 = 386\)。这意味着，考虑到9个智能体和15个不可动物体的情况下，每个智能体的观测向量将会是386维的。这个维度的计算反映了智能体需要处理的信息量，包括其在环境中的位置、与其他智能体的相对位置，以及与环境中不可动物体的相对位置。
        
        # 读取智能体相关信息CSV文件到DataFrame
        self.csv_path = 'C:\\Users\\ALIENWARE\\Desktop\\AI_Biuldings\\Micro-Update Study\\mission.csv'
        # 标准化压缩比例
        self.standard_opening = 32.0
        self.standard_depth = 14.0 #原为16
        #self.agents_sizes = self.extract_flat_dimensions_from_csv()
        # ImmovableObjects是柱子等不可移动物体的位置，在GH中获得。        
        #self.ImmovableObjects = self.standardize_positions([-16, -8, -16, 0, -16, 8, -8, -8, -8, 0, -8, 8, 0, -8, 0, 0, 0, 8, 8, -8, 8, 0, 8, 8, 16, -8, 16, 0, 16, 8])
        self.ImmovableObjects = self.standardize_positions([-16, -8, -16, 0, -16, 6, -8, -8, -8, 0, -8, 6, 0, -8, 0, 0, 0, 6, 8, -8, 8, 0, 8, 6, 16, -8, 16, 0, 16, 6])
        single_agent_obs_space = spaces.Box(low=np.array([-1]*386),
                                                high=np.array([1]*386), dtype=np.float32)

        # 为多个智能体定义观察空间列表
        self.observation_space = [single_agent_obs_space]*self.num_agents
        self.observation_spaces = {f"agent_{i}": space for i, space in enumerate(self.observation_space)}
        self.condition = threading.Condition()
        self.data_received = False
        self.data_send = False
        self.agents_rewards = [0 for _ in range(self.num_agents)]
        self.actions_list = [0 for _ in range(4 * self.num_agents)]
        self.initialized_positions = self.initialize_positions()
        self.corners = [0 for _ in range(8 * self.num_agents)]
        self.agent_coords = [0 for _ in range(2 * self.num_agents)]
        # 定义 obs_shape_n 和 act_shape_n
        # 定义 obs_shape_n 和 act_shape_n
        # obs_shape_n将是一个形如[(48,), (48,)]的列表，表示两个智能体的观察空间都是48D向量。
        # 同样，act_shape_n将是一个形如[(2,), (2,)]的列表，表示两个智能体的动作空间都是2D向量。
        self.obs_shape_n = [np.prod(agent_obs_space.shape) for agent_obs_space in self.observation_space]
        self.act_shape_n = [np.prod(agent_action_space.shape) for agent_action_space in self.action_space]
        # 初始化 agents_obs
        
        #self.agents_obs = [np.array([0, 0, 0.75, 0.75, -0.5, -0.5]), np.array([0, 0, 0.25, 0.25, 0.5, 0.5])]
        #self.dataSend = [0.5, 0.5]*self.num_agents
        # 初始化 dataSend
        self.dataSend = self.initialize_positions()   

        # UDP 初始化
        self.UDP_IP = "127.0.0.1"
        self.SEND_PORT = 12345
        self.RECEIVE_PORT = 54321
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receive_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receive_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)  # 设置缓冲区大小
        self.receive_sock.bind((self.UDP_IP, self.RECEIVE_PORT))
        self.dataReceive_lock = threading.Lock()  # 线程锁，用于确保状态的线程安全性
        self.dataSend_lock = threading.Lock()
        self.reset()

        # 启动发送和监听线程
        self.send_thread = threading.Thread(target=self.send_message)
        self.listen_thread = threading.Thread(target=self.listen_for_responses)
        self.send_thread.start()
        self.listen_thread.start()
         
    def split_combined_input(self, combined_input):
        # 奖励值数量等于智能体数量
        rewards = combined_input[:self.num_agents]
        # 假设每个智能体的坐标是二维的（x, y），则每个智能体有两个坐标值
        # 因此，智能体坐标的总长度是智能体数量乘以2
        agent_coords_length = self.num_agents * 2
        # 智能体坐标紧跟在奖励值之后
        agent_coords = combined_input[self.num_agents:self.num_agents + agent_coords_length]
        # 剩余部分是角点坐标
        corners = combined_input[self.num_agents + agent_coords_length:]
        
        return rewards, agent_coords, corners


    def send_message(self):
        while True:
            with self.condition:
                message = str(self.dataSend)
                self.send_sock.sendto(message.encode(), (self.UDP_IP, self.SEND_PORT))
                self.data_send = True
                self.data_received = False
                self.condition.notify_all()

    def listen_for_responses(self):
        while True:
            data, addr = self.receive_sock.recvfrom(2048)
            with self.condition:
                self.dataReceive = data.decode()
                self.data_received = True
                message = json.loads(self.dataReceive)
                self.agents_rewards_receive, self.agent_coords, self.corners = self.split_combined_input(message)
                self.condition.notify_all()
                
    def standardize_positions(self, positions):
        """
        标准化位置列表，使其在-1到1的范围内。
        :param positions: 包含多个坐标的扁平一维列表，格式为[x1, y1, x2, y2, ...]。
        :param standard_opening: 标准化的宽度基准。
        :param standard_depth: 标准化的高度基准。
        :return: 标准化后的位置列表。
        """
        standard_opening = self.standard_opening
        standard_depth = self.standard_depth
        max_opening = standard_opening / 2.0
        max_depth = standard_depth / 2.0

        # 标准化位置
        standardized_positions = []
        for i in range(0, len(positions), 2):
            x, y = positions[i], positions[i + 1]
            standardized_x = x / max_opening
            standardized_y = y / max_depth
            standardized_positions.extend([standardized_x, standardized_y])

        return standardized_positions
                
    # def extract_flat_dimensions_from_csv(self):
    #     # 读取CSV文件到DataFrame
    #     df = pd.read_csv(self.csv_path)
    #     # 初始化扁平列表
    #     flat_list = []
    #     # 遍历DataFrame的每一行
    #     for _, row in df.iterrows():
    #         # 标准化opening和depth
    #         opening = row['opening']
    #         depth = row['depth']
    #         # 遍历每个功能的数量
    #         for _ in range(int(row['num'])):
    #             # 将长度和宽度添加到列表
    #             flat_list.extend([opening, depth])
    #     flat_list = self.standardize_positions(flat_list)
        
    #     return flat_list
    
    
    def calculate_corners_flat(self, positions):
        """
        根据智能体的中心点坐标和大小计算每个智能体的四个角的坐标，并以扁平一维列表形式输出。
        :param positions: 包含多个智能体中心点坐标的扁平一维列表，格式为[x1, y1, x2, y2, ...]。
        :param sizes: 包含多个智能体大小的扁平一维列表，格式为[o1, d1, o2, d2, ...]。
        :return: 扁平化的一维列表，包含每个智能体四个角的坐标。
        """
        sizes = self.agents_sizes
        num_agents = self.num_agents
        corners_flat = []

        for i in range(num_agents):
            x, y = positions[2*i], positions[2*i + 1]
            opening, depth = sizes[2*i], sizes[2*i + 1]

            # 计算四个角的坐标
            top_left = (x - opening/2, y + depth/2)
            top_right = (x + opening/2, y + depth/2)
            bottom_left = (x - opening/2, y - depth/2)
            bottom_right = (x + opening/2, y - depth/2)

            # 将四个角的坐标扁平化并添加到总列表中
            corners_flat.extend([top_left, top_right, bottom_left, bottom_right])

        # 将嵌套的元组扁平化为一维列表
        corners_flat = [coord for corner in corners_flat for coord in corner]

        return corners_flat




    def create_observations(self, agent_positions, corners, object_positions):
        num_agents = self.num_agents
        objects_array = np.array(object_positions).reshape(-1, 2)
        observations = []

        # 假设 agent_positions 是一个扁平列表，其中坐标以 xyxy 交替的方式出现
        # corners 同样是扁平的，但每个智能体有四个角点，每个角点由一对 x, y 坐标表示

        for agent_idx in range(num_agents):
            # 获取当前智能体的坐标
            # 每个智能体的坐标占用两个位置 (x, y)，因此索引是 agent_idx * 2
            agent_coord_index = agent_idx * 2
            agent_coord = np.array(agent_positions[agent_coord_index:agent_coord_index + 2])

            # 获取当前智能体的四角坐标
            # 每个智能体的四角坐标占用8个位置 (x1, y1, x2, y2, x3, y3, x4, y4)
            agent_corners_index = agent_idx * 8
            agent_corners = np.array(corners[agent_corners_index:agent_corners_index + 8]).reshape(-1, 2)

            # 计算相对于其他智能体四角的位置和相对于不可动物体的位置的代码保持不变
            relative_agent_corners = []
            for other_idx in range(num_agents):
                if other_idx != agent_idx:
                    other_corners = np.array(corners[other_idx * 8: (other_idx + 1) * 8]).reshape(-1, 2)
                    for agent_corner in agent_corners:
                        for other_corner in other_corners:
                            relative_positions = agent_corner - other_corner
                            relative_agent_corners.extend(relative_positions.flatten())

            relative_object_positions = []
            for object_position in objects_array:
                for corner in agent_corners:
                    relative_positions = corner - object_position
                    relative_object_positions.extend(relative_positions.flatten())

            # 将智能体坐标、自身四角坐标、相对于其他智能体四角的位置和相对于不可动物体的位置合并到一个数组中
            obs = np.concatenate([agent_coord.flatten(), agent_corners.flatten(), np.array(relative_agent_corners), np.array(relative_object_positions)])
            observations.append(obs)

        return np.array(observations)

   
    
    
    def update_positions(self, current_positions, velocities):
        """
        根据四个方向的速度更新多个智能体的位置，并确保智能体的整个矩形在-1到1的范围内。
        同时，将输出的位置精确到一位小数点。
        :param current_positions: 包含多个智能体位置的扁平一维列表，格式为[x1, y1, x2, y2, ...]。
        :param velocities: 包含多个智能体四个方向速度的扁平一维列表，格式为[u1, d1, l1, r1, u2, d2, l2, r2, ...]。
        :param sizes: 包含多个智能体尺寸的扁平一维列表，格式为[w1, h1, w2, h2, ...]。
        :return: 更新后的位置列表，每个位置精确到一位小数点。
        """
        sizes = self.agents_sizes
        num_agents = self.num_agents
        positions_array = np.array(current_positions).reshape(num_agents, 2)
        velocities_array = np.array(velocities).reshape(num_agents, 4)
        sizes_array = np.array(sizes).reshape(num_agents, 2)

        # 更新位置
        for i in range(num_agents):
            up, down, left, right = velocities_array[i]
            width, height = sizes_array[i]

            new_x = positions_array[i, 0] + right - left
            new_y = positions_array[i, 1] + up - down

            # 先进行四舍五入
            new_x = np.round(new_x, 1)
            new_y = np.round(new_y, 1)

            # 确保矩形的整个部分在边界内
            half_width, half_height = width / 2, height / 2
            new_x = np.clip(new_x, -1 + half_width, 1 - half_width)
            new_y = np.clip(new_y, -1 + half_height, 1 - half_height)

            positions_array[i, 0] = new_x
            positions_array[i, 1] = new_y

        return positions_array.flatten().tolist()


    def initialize_positions(self):
        positions = []

        while len(positions) < self.num_agents * 2:
            # 生成随机中心点位置，同时考虑尺寸以保证矩形区域在边界内，并确保位置保留4位小数
            x = round(random.uniform(-1, 1), 4)
            y = round(random.uniform(-1, 1), 4)

            # 检查是否与已有位置重合
            if not any(x == pos[0] and y == pos[1] for pos in zip(positions[::2], positions[1::2])):
                positions.extend([x, y])

        return positions

    def reset(self): 
       
        self.initialized_positions = self.initialize_positions()

        # 发送初始化信号给grasshopper
        self.dataSend = self.initialized_positions + self.actions_list + [1] 
                    
        # 重置 agents_obs 为默认初始观察值
        self.agents_obs = self.create_observations(self.agent_coords, self.corners, self.ImmovableObjects)
        obs = {f"agent_{i}": obs for i, obs in enumerate(self.agents_obs)}
        # 重置步骤计数器
        self.episode_steps = 0    
        return obs
      
    def step(self, actions):
        # 初始化字典
        dones = {}
        infos = {}
        obs_dict = {}
        rewards_dict = {}

        # 等待 listen_for_responses 接收数据
        with self.condition:
            while not self.data_received:
                self.condition.wait()  

        # 转换 actions 从字典到列表
        actions_list = [action for action in actions.values()]
        actions_list = [float(item) for sublist in actions_list for item in sublist]

        # 通过速度计算更新智能体位置信息
        self.dataSend = self.initialized_positions + actions_list + [0]
        self.agents_obs = self.create_observations(self.agent_coords, self.corners, self.ImmovableObjects)
        self.agents_rewards = self.agents_rewards_receive
        
        # 检查是否达到成功或失败的状态
        success_done = sum(self.agents_rewards) > 12

        if success_done:
            done = True
            rewards = [100 for _ in range(self.num_agents)]

        else:
            done = False
            rewards = self.agents_rewards
            
        # 增加步骤计数
        self.episode_steps += 1
        
        # 将数据转换为字典格式
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            obs_dict[agent_id] = self.agents_obs[i]
            rewards_dict[agent_id] = rewards[i]
            dones[agent_id] = done
            infos[agent_id] = {}

        return obs_dict, rewards_dict, dones, infos


    def close(self):
        self.send_sock.close()
        self.receive_sock.close()
        self.send_thread.join()
        self.listen_thread.join()

        
class mpe_wrapper_for_GrasshopperEnv(gym.Wrapper):
    def __init__(self, env=None, continuous_actions=False):
        gym.Wrapper.__init__(self, env)
        self.continuous_actions = continuous_actions
        self.observation_space = list(self.observation_spaces.values())
        self.action_space = list(self.action_spaces.values())
        assert len(self.observation_space) == len(self.action_space)
        self.n = len(self.observation_space)
        self.agents_name = list(self.observation_spaces.keys())
        self.obs_shape_n = [
            self.get_shape(self.observation_space[i]) for i in range(self.n)
        ]
        self.act_shape_n = [
            self.get_shape(self.action_space[i]) for i in range(self.n)
        ]

    def get_shape(self, input_space):
        """
        Args:
            input_space: environment space

        Returns:
            space shape
        """
        if (isinstance(input_space, spaces.Box)):
            if (len(input_space.shape) == 1):
                return input_space.shape[0]
            else:
                return input_space.shape
        elif (isinstance(input_space, spaces.Discrete)):
            return input_space.n
        else:
            print('[Error] shape is {}, not Box or Discrete'.format(
                input_space.shape))
            raise NotImplementedError

    def reset(self):
        obs = self.env.reset()
        return list(obs.values())

    def step(self, actions):
        actions_dict = dict()
        for i, act in enumerate(actions):
            agent = self.agents_name[i]
            if self.continuous_actions:
                assert np.all(((act<=1.0 + 1e-3), (act>=-1.0 - 1e-3))), \
                    'the action should be in range [-1.0, 1.0], but got {}'.format(act)
                high = self.action_space[i].high
                low = self.action_space[i].low
                mapped_action = low + (act - (-1.0)) * ((high - low) / 2.0)
                mapped_action = np.clip(mapped_action, low, high)
                actions_dict[agent] = mapped_action
            else:
                actions_dict[agent] = np.argmax(act)
        obs, reward, done, info = self.env.step(actions_dict)
        return list(obs.values()), list(reward.values()), list(
            done.values()), list(info.values())

        
def GHenv(continuous_actions=True):
    env = GrasshopperEnv()
    env = mpe_wrapper_for_GrasshopperEnv(env, continuous_actions = continuous_actions)
    return env


"""
env = GHenv()
for _ in range(30):
    env.reset()
    done = False  # 初始化为False
    while not done:
        actions = [space.sample() for space in env.action_space]
        print(actions)  
        obs, rewards, dones, infos = env.step(actions)  
        print(obs, rewards, dones, infos)
        print("---------------------------------------")
        print(len(obs[0]))
        print("---------------------------------------")
        done = any(dones)
    print("Episode done\n")
"""