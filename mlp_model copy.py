# mlp_model.py
import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class MAModel(parl.Model):
    def __init__(self,
                 obs_dim,          # 单个智能体的观测空间形状，例如 (3, 64, 64)
                 act_dim,          # 单个智能体的动作空间维度，例如 2
                 obs_shape_n,      # 所有智能体的观测空间形状列表
                 act_shape_n,      # 所有智能体的动作空间维度列表
                 continuous_actions=False):
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
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size)
        self.drop1 = nn.Dropout(p=self.dropout_probability)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size2)
        self.drop2 = nn.Dropout(p=self.dropout_probability)
        self.fc3 = nn.Linear(self.hidden_size2, self.act_dim)

        if self.continuous_actions:
            self.std_fc = nn.Linear(self.hidden_size2, self.act_dim)

    def forward(self, obs):
        """
        前向传播，计算动作分布或确定性动作。
        
        Args:
            obs (paddle.Tensor): 观测，形状为 [batch_size, C, H, W]。
        
        Returns:
            paddle.Tensor: 动作或动作分布的参数。
        """
        # 仅提取第三个通道
        obs = obs[:, 2, :, :]  # shape: [batch_size, H, W]
        x = obs.reshape([obs.shape[0], -1])  # 展平，shape: [batch_size, H*W]

        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        means = self.fc3(x)

        if self.continuous_actions:
            means = F.tanh(means)  # 确保输出在 [-1, 1] 范围内
            act_std = self.std_fc(x)
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
        self.fc1 = nn.Linear((self.input_dim + act_shape_n[0]) * self.n, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2, 1)

    def forward(self, obs_n, act_n):
        """
        前向传播，计算 Q 值。
        
        Args:
            obs_n (list of paddle.Tensor): 包含所有智能体观测的列表，每个元素形状为 [batch_size, C, H, W]。
            act_n (list of paddle.Tensor): 包含所有智能体动作的列表，每个元素形状为 [batch_size, act_dim]。
        
        Returns:
            paddle.Tensor: Q 值，形状为 [batch_size, 1]。
        """
        features = []
        for i in range(self.n):
            # 仅提取第三个通道
            obs = obs_n[i][:, 2, :, :]  # shape: [batch_size, H, W]
            x = obs.reshape([obs.shape[0], -1])  # 展平，shape: [batch_size, H*W]
            act = act_n[i]  # shape: [batch_size, act_dim]
            combined = paddle.concat([x, act], axis=1)  # shape: [batch_size, H*W + act_dim]
            features.append(combined)

        # 拼接所有智能体的特征和动作
        combined_features = paddle.concat(features, axis=1)  # shape: [batch_size, n*(H*W + act_dim)]

        # 通过全连接层
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
