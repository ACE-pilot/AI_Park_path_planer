# hyper_model.py
import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import logging

# 定义观测范围超参数
ELEVATION_OBS_SIZE = 11  # 表示 11x11 的观测范围

# 配置日志
logging.basicConfig(level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("error.log"),
                              logging.StreamHandler()])

class MAModel(parl.Model):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 obs_shape_n,
                 act_shape_n,
                 continuous_actions=False):
        super(MAModel, self).__init__()
        
        if not isinstance(obs_shape_n, list) or not all(isinstance(shape, tuple) for shape in obs_shape_n):
            raise ValueError("obs_shape_n 必须是包含元组的列表，例如 [(58,), (58,), (58,)]")
        if not isinstance(act_shape_n, list) or not all(isinstance(shape, int) for shape in act_shape_n):
            raise ValueError("act_shape_n 必须是整数的列表，例如 [2, 2, 2]")
        
        # 计算 critic_in_dim
        critic_in_dim = sum([shape[0] for shape in obs_shape_n]) + sum(act_shape_n)
        logging.info(f"critic_in_dim: {critic_in_dim}")
        
        if isinstance(obs_dim, tuple):
            obs_dim = obs_dim[0]
        if isinstance(act_dim, tuple):
            act_dim = act_dim[0]
        
        self.actor_model = ActorModel(obs_dim, act_dim, continuous_actions)
        self.critic_model = CriticModel(critic_in_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, act):
        return self.critic_model(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, obs_dim, act_dim, continuous_actions=False):
        super(ActorModel, self).__init__()
        self.continuous_actions = continuous_actions
        
        # 检查 obs_dim 是否足够大，确保可以减去 ELEVATION_OBS_SIZE**2
        print(obs_dim)
        if obs_dim <= ELEVATION_OBS_SIZE**2:
            raise ValueError(f"obs_dim ({obs_dim}) 必须大于 ELEVATION_OBS_SIZE**2 ({ELEVATION_OBS_SIZE**2})，以确保输入尺寸有效。请调整 ELEVATION_OBS_SIZE 或 obs_dim。")
        
        # MLP 部分处理常规观测
        self.mlp_fc1 = nn.Linear(obs_dim - ELEVATION_OBS_SIZE**2, 64, weight_attr=paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform()))
        self.mlp_fc2 = nn.Linear(64, 64, weight_attr=paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform()))
        
        # CNN 处理高程矩阵
        self.conv1 = nn.Conv2D(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2D(kernel_size=2)  # 新增池化层
        self.flatten = nn.Flatten()
        
        # 新的合并隐藏层，用于融合 MLP 和 CNN 特征
        self.shared_fc = nn.Linear(64 + 32 * (ELEVATION_OBS_SIZE // 2) * (ELEVATION_OBS_SIZE // 2), 64, weight_attr=paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform()))
        self.fc_out = nn.Linear(64, act_dim, weight_attr=paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform()))

        if self.continuous_actions:
            self.std_fc = nn.Linear(64, act_dim, weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))

    def forward(self, obs):
        # MLP 处理常规观测
        mlp_input = obs[:, :-ELEVATION_OBS_SIZE**2]
        mlp_hid1 = F.relu(self.mlp_fc1(mlp_input))
        mlp_hid2 = F.relu(self.mlp_fc2(mlp_hid1))

        # CNN 处理高程矩阵，增加池化层减少参数量
        elevation_matrix = obs[:, -ELEVATION_OBS_SIZE**2:].reshape([-1, 1, ELEVATION_OBS_SIZE, ELEVATION_OBS_SIZE])
        conv_hid1 = F.relu(self.conv1(elevation_matrix))
        conv_hid2 = F.relu(self.pool(self.conv2(conv_hid1)))
        cnn_out = self.flatten(conv_hid2)

        # 合并 MLP 和 CNN 特征
        combined_input = paddle.concat([mlp_hid2, cnn_out], axis=1)
        combined_hid = F.relu(self.shared_fc(combined_input))  # 新的共享层
        means = self.fc_out(combined_hid)
        
        if self.continuous_actions:
            act_std = self.std_fc(combined_hid)
            act_std = paddle.clip(act_std, min=1e-6, max=1.0)  # 保证标准差处于合理范围
            return means, act_std
        return means


class CriticModel(parl.Model):
    def __init__(self, critic_in_dim):
        super(CriticModel, self).__init__()
        hid1_size = 64
        hid2_size = 64
        out_dim = 1
        self.fc1 = nn.Linear(
            critic_in_dim,
            hid1_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.fc2 = nn.Linear(
            hid1_size,
            hid2_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.fc3 = nn.Linear(
            hid2_size,
            out_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))

    def forward(self, obs_n, act_n):
        if not isinstance(obs_n, list) or not isinstance(act_n, list):
            raise ValueError("obs_n 和 act_n 必须是列表类型")
        if len(obs_n) != len(act_n):
            raise ValueError("obs_n 和 act_n 的长度必须相同")
        
        for i in range(len(obs_n)):
            if not isinstance(obs_n[i], paddle.Tensor) or not isinstance(act_n[i], paddle.Tensor):
                raise ValueError("每个 obs 和 act 必须是 Paddle 张量")
        
        # 拼接所有智能体的观测和动作
        inputs = paddle.concat(obs_n + act_n, axis=1)
        hid1 = F.relu(self.fc1(inputs))
        hid2 = F.relu(self.fc2(hid1))
        Q = self.fc3(hid2)
        Q = paddle.squeeze(Q, axis=1)
        return Q