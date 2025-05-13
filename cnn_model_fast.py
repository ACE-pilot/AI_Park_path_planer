import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class MAModel(parl.Model):
    def __init__(self,
                 obs_dim,          # 单个智能体的观测空间形状，例如 (3, 128, 128)
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
    def __init__(self, obs_shape, act_dim, continuous_actions=False):
        super(ActorModel, self).__init__()
        self.continuous_actions = continuous_actions
        print(f"-----ActorModel obs_shape------: {obs_shape}")

        # 将观测展平
        self.flatten = nn.Flatten()

        # 仅使用前128个数
        self.obs_reduction = nn.Linear(paddle.prod(paddle.to_tensor(obs_shape)).item(), 128)

        # 使用极小的线性层替代卷积层
        hidden_size = 8  # 隐藏层大小为个位数
        self.fc1 = nn.Linear(in_features=128, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=act_dim)

        if self.continuous_actions:
            self.std_fc = nn.Linear(in_features=hidden_size, out_features=act_dim)

    def forward(self, obs):
        x = self.flatten(obs)
        # 仅选择前128个数
        x = x[:, :128]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.continuous_actions:
            means = F.tanh(self.fc3(x))  # 输出范围 [-1, 1]
            act_std = F.softplus(self.std_fc(x))  # 确保标准差为正
            return means, act_std
        else:
            means = self.fc3(x)
            return means


class CriticModel(parl.Model):
    def __init__(self, obs_shape_n, act_shape_n, continuous_actions=False):
        super(CriticModel, self).__init__()
        self.continuous_actions = continuous_actions
        self.n = len(obs_shape_n)  # 智能体数量

        # 将每个智能体的观测展平
        self.flatten = nn.Flatten()

        # 仅使用前128个数
        self.obs_reduction = nn.Linear(paddle.prod(paddle.to_tensor(obs_shape_n[0])).item(), 128)

        single_act_size = act_shape_n[0]
        total_input_size = self.n * (128 + single_act_size)  # 使用前128个数

        # 使用极简的线性层
        hidden_size = 8  # 极小隐藏层
        self.fc1 = nn.Linear(in_features=total_input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, obs_n, act_n):
        features = []
        for i in range(self.n):
            obs = obs_n[i]
            act = act_n[i]
            x = self.flatten(obs)
            # 仅选择前128个数
            x = x[:, :128]
            feature = paddle.concat([x, act], axis=1)
            features.append(feature)
        combined = paddle.concat(features, axis=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
