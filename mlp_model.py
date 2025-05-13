# mlp_model.py
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
                 obs_dim,          # 单个智能体的观测空间维度，例如 10
                 act_dim,          # 单个智能体的动作空间维度，例如 2
                 obs_shape_n,      # 所有智能体的观测空间维度列表（整数列表，例如 [10, 10, 10]）
                 act_shape_n,      # 所有智能体的动作空间维度列表（整数列表，例如 [2, 2, 2]）
                 continuous_actions=True):
        super(MAModel, self).__init__()
        self.actor_model = ActorModel(obs_dim, act_dim, continuous_actions)
        self.critic_model = CriticModel(obs_shape_n, act_shape_n, continuous_actions)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs_n, act_n):
        return self.critic_model(obs_n, act_n)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, obs_dim, act_dim, continuous_actions=True):
        super(ActorModel, self).__init__()
        self.continuous_actions = continuous_actions

        # 确保 obs_dim 是整数而非元组
        if isinstance(obs_dim, tuple):
            obs_dim = obs_dim[0]

        # 定义 MLP 超参数
        self.hidden_size = 64
        self.hidden_size2 = 64
        self.act_dim = act_dim

        # 定义全连接层
        self.fc1 = nn.Linear(obs_dim, self.hidden_size, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform()))
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size2, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform()))
        self.fc3 = nn.Linear(self.hidden_size2, self.act_dim, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform()))

        if self.continuous_actions:
            self.std_fc = nn.Linear(self.hidden_size2, self.act_dim, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform()))

    def forward(self, obs):
        hid1 = F.relu(self.fc1(obs))
        hid2 = F.relu(self.fc2(hid1))
        means = self.fc3(hid2)
        if self.continuous_actions:
            means = F.tanh(means)  # 确保输出在 [-1, 1] 范围内
            act_std = F.softplus(self.std_fc(hid2)) + 1e-6  # 使用 Softplus 确保 act_std 为正数，并添加小常数防止为零
            return means, act_std
        return means


class CriticModel(parl.Model):
    def __init__(self, obs_shape_n, act_shape_n, continuous_actions=True):
        super(CriticModel, self).__init__()
        self.continuous_actions = continuous_actions
        self.n = len(obs_shape_n)  # 智能体数量

        # 确保 obs_shape_n 是整数列表
        obs_dim_list = [shape[0] for shape in obs_shape_n]  # 从元组中提取整数
        act_dim_list = act_shape_n  # 已经是整数列表

        # 计算所有智能体的观测和动作的总维度
        self.input_dim = sum(obs_dim_list) + sum(act_dim_list)  # 总输入维度

        # 定义 MLP 超参数
        self.hidden_size = 64
        self.hidden_size2 = 64

        # 定义全连接层
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform()))
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size2, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform()))
        self.fc3 = nn.Linear(self.hidden_size2, 1, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform()))

    def forward(self, obs_n, act_n):
        """
        前向传播函数
        obs_n: list of [batch_size, obs_dim] tensors for each agent
        act_n: list of [batch_size, act_dim] tensors for each agent
        """
        features = []
        for i in range(self.n):
            # 每个智能体的观测和动作
            obs = obs_n[i]  # shape: [batch_size, obs_dim]
            act = act_n[i]  # shape: [batch_size, act_dim]
            combined = paddle.concat([obs, act], axis=1)  # shape: [batch_size, obs_dim + act_dim]
            features.append(combined)

        # 拼接所有智能体的特征和动作
        combined_features = paddle.concat(features, axis=1)  # shape: [batch_size, n*(obs_dim + act_dim)]

        # 通过全连接层
        x = F.relu(self.fc1(combined_features))  # Layer 1
        x = F.relu(self.fc2(x))  # Layer 2
        q = self.fc3(x)  # 输出 Q 值
        q = paddle.squeeze(q, axis=1)
        return q

# 注意：不需要修改 MADDPGWrapper 代码
