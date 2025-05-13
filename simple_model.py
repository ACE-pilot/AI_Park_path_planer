# simple_model.py
import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import logging

# 配置日志
logging.basicConfig(level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("error.log"),
                              logging.StreamHandler()])

class MAModel(parl.Model):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 obs_shape_n,      # 新增参数
                 act_shape_n,      # 新增参数
                 continuous_actions=False):
        super(MAModel, self).__init__()
        
        # 确保 obs_shape_n 和 act_shape_n 是正确的格式
        if not isinstance(obs_shape_n, list) or not all(isinstance(shape, tuple) for shape in obs_shape_n):
            raise ValueError("obs_shape_n 必须是包含元组的列表，例如 [(58,), (58,), (58,)]")
        if not isinstance(act_shape_n, list) or not all(isinstance(shape, int) for shape in act_shape_n):
            raise ValueError("act_shape_n 必须是整数的列表，例如 [2, 2, 2]")
        
        # 计算 critic_in_dim
        critic_in_dim = sum([shape[0] for shape in obs_shape_n]) + sum(act_shape_n)
        logging.info(f"critic_in_dim: {critic_in_dim}")  # 添加日志以确认计算结果
        
        # 确保 obs_dim 和 act_dim 是整数而非元组
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
        hid1_size = 64
        hid2_size = 64
        self.fc1 = nn.Linear(
            obs_dim,
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
            act_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        if self.continuous_actions:
            std_hid_size = 64
            self.std_fc = nn.Linear(
                std_hid_size,
                act_dim,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()))

    def forward(self, obs):
        hid1 = F.relu(self.fc1(obs))
        hid2 = F.relu(self.fc2(hid1))
        means = self.fc3(hid2)
        if self.continuous_actions:
            act_std = self.std_fc(hid2)
            return (means, act_std)
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
        # 确保 obs_n 和 act_n 是列表，并且包含正确数量的元素
        if not isinstance(obs_n, list) or not isinstance(act_n, list):
            raise ValueError("obs_n 和 act_n 必须是列表类型")
        if len(obs_n) != len(act_n):
            raise ValueError("obs_n 和 act_n 的长度必须相同")
        
        # 确保每个 obs 和 act 都是 Paddle 张量
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

# 注意：不需要修改 MADDPGWrapper 代码
