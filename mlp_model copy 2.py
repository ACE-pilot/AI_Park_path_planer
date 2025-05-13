# mlp_model.py
import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import logging
import numpy as np

# 配置日志
logging.basicConfig(level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("error.log"),
                              logging.StreamHandler()])

class MAModel(parl.Model):
    def __init__(self,
                 obs_dim,          # 单个智能体的观测空间形状，例如 (3, 64, 64)
                 act_dim,          # 单个智能体的动作空间维度，例如 2
                 obs_shape_n,      # 所有智能体的观测空间形状列表
                 act_shape_n,      # 所有智能体的动作空间维度列表
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
    def __init__(self, obs_shape, act_dim, continuous_actions=True):
        super(ActorModel, self).__init__()
        self.continuous_actions = continuous_actions

        # 提取第三个通道的数据
        self.input_dim = obs_shape[1] * obs_shape[2]  # H * W

        # 定义 MLP 超参数
        self.hidden_size = 256
        self.hidden_size2 = 128
        self.dropout_probability = 0.3
        self.act_dim = act_dim

        # 定义全连接层
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.bn1 = nn.BatchNorm1D(self.hidden_size)
        self.drop1 = nn.Dropout(p=self.dropout_probability)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size2, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.bn2 = nn.BatchNorm1D(self.hidden_size2)
        self.drop2 = nn.Dropout(p=self.dropout_probability)
        self.fc3 = nn.Linear(self.hidden_size2, self.act_dim, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform()))

        if self.continuous_actions:
            self.std_fc = nn.Linear(self.hidden_size2, self.act_dim, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform()))

    def forward(self, obs):
        # 仅提取第三个通道
        obs = obs[:, 2, :, :]  # shape: [batch_size, H, W]
        x = obs.reshape([obs.shape[0], -1])  # 展平，shape: [batch_size, H*W]

        x = F.leaky_relu(self.bn1(self.fc1(x)))  # 使用 Leaky ReLU
        if paddle.isnan(x).any():
            logging.error("NaN detected after fc1 (Actor)")
            logging.error(f"x after fc1: {x}")
        x = self.drop1(x)
        if paddle.isnan(x).any():
            logging.error("NaN detected after drop1 (Actor)")
            logging.error(f"x after drop1: {x}")

        x = F.leaky_relu(self.bn2(self.fc2(x)))  # 使用 Leaky ReLU
        if paddle.isnan(x).any():
            logging.error("NaN detected after fc2 (Actor)")
            logging.error(f"x after fc2: {x}")
        x = self.drop2(x)
        if paddle.isnan(x).any():
            logging.error("NaN detected after drop2 (Actor)")
            logging.error(f"x after drop2: {x}")

        means = self.fc3(x)
        if paddle.isnan(means).any():
            logging.error("NaN detected after fc3 (Actor)")
            logging.error(f"means: {means}")

        if self.continuous_actions:
            means = F.tanh(means)  # 确保输出在 [-1, 1] 范围内
            if paddle.isnan(means).any():
                logging.error("NaN detected after tanh (Actor)")
                logging.error(f"means after tanh: {means}")
            act_std = F.softplus(self.std_fc(x)) + 1e-6  # 使用 Softplus 确保 act_std 为正数，并添加小常数防止为零
            if paddle.isnan(act_std).any():
                logging.error("NaN detected after std_fc (Actor)")
                logging.error(f"act_std: {act_std}")
            return means, act_std
        return means


class CriticModel(parl.Model):
    def __init__(self, obs_shape_n, act_shape_n, continuous_actions=True):
        super(CriticModel, self).__init__()
        self.continuous_actions = continuous_actions
        self.n = len(obs_shape_n)  # 智能体数量

        # 提取第三个通道的数据
        self.input_dim = obs_shape_n[0][1] * obs_shape_n[0][2]  # H * W

        # 定义 MLP 超参数
        self.hidden_size = 256
        self.hidden_size2 = 128

        # 定义全连接层
        # 每个智能体的输入是 (H*W) + act_dim
        self.fc1 = nn.Linear((self.input_dim + act_shape_n[0]) * self.n, self.hidden_size, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.bn1 = nn.BatchNorm1D(self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size2, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.bn2 = nn.BatchNorm1D(self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2, 1, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform()))

    def forward(self, obs_n, act_n):
        features = []
        for i in range(self.n):
            # 仅提取第三个通道
            obs = obs_n[i][:, 2, :, :]  # shape: [batch_size, H, W]
            x = obs.reshape([obs.shape[0], -1])  # 展平，shape: [batch_size, H*W]
            act = act_n[i]  # shape: [batch_size, act_dim]
            combined = paddle.concat([x, act], axis=1)  # shape: [batch_size, H*W + act_dim]
            if paddle.isnan(combined).any():
                logging.error(f"NaN detected in combined features for agent {i} (Critic)")
                logging.error(f"combined for agent {i}: {combined}")
            features.append(combined)

        # 拼接所有智能体的特征和动作
        combined_features = paddle.concat(features, axis=1)  # shape: [batch_size, n*(H*W + act_dim)]
        if paddle.isnan(combined_features).any():
            logging.error("NaN detected in combined_features (Critic)")
            logging.error(f"combined_features: {combined_features}")

        # 通过全连接层
        x = F.leaky_relu(self.bn1(self.fc1(combined_features)))  # 使用 Leaky ReLU
        if paddle.isnan(x).any():
            logging.error("NaN detected after fc1 (Critic)")
            logging.error(f"x after fc1: {x}")
        x = F.leaky_relu(self.bn2(self.fc2(x)))  # 使用 Leaky ReLU
        if paddle.isnan(x).any():
            logging.error("NaN detected after fc2 (Critic)")
            logging.error(f"x after fc2: {x}")
        q = self.fc3(x)
        if paddle.isnan(q).any():
            logging.error("NaN detected after fc3 (Critic)")
            logging.error(f"q: {q}")
        return q
