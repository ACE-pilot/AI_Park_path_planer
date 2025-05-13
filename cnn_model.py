# cnn_model.py
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
    def __init__(self, obs_shape, act_dim, continuous_actions=False):
        super(ActorModel, self).__init__()
        self.continuous_actions = continuous_actions
        #print(f"-----ActorModel obs_shape------: {obs_shape}")

        # 定义 CNN 超参数
        self.filter_size = 5
        self.num_filters = [32, 64, 64]
        self.stride = 1
        self.pool_size = 2
        self.dropout_probability = [0.3, 0.2]
        self.hidden_size = 128
        self.act_dim = act_dim

        # 定义 CNN 层，添加 padding 以保持输出尺寸
        self.conv1 = nn.Conv2D(in_channels=obs_shape[0], out_channels=self.num_filters[0],
                               kernel_size=self.filter_size, stride=self.stride, padding=2)
        self.pool1 = nn.MaxPool2D(kernel_size=self.pool_size, stride=self.pool_size)

        self.conv2 = nn.Conv2D(in_channels=self.num_filters[0], out_channels=self.num_filters[1],
                               kernel_size=self.filter_size, stride=self.stride, padding=2)
        self.pool2 = nn.MaxPool2D(kernel_size=self.pool_size, stride=self.pool_size)

        self.conv3 = nn.Conv2D(in_channels=self.num_filters[1], out_channels=self.num_filters[2],
                               kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2D(kernel_size=self.pool_size, stride=self.pool_size)

        # 计算全连接层的输入大小
        self.fc_input_size = self._get_conv_output_size(obs_shape)
        #print(f"ActorModel fc_input_size: {self.fc_input_size}, type: {type(self.fc_input_size)}")
        assert isinstance(self.fc_input_size, int), "fc_input_size 应该是一个整数"

        # 定义全连接层
        self.fc1 = nn.Linear(self.fc_input_size, self.hidden_size * 2)
        self.drop1 = nn.Dropout(p=self.dropout_probability[0])
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.drop2 = nn.Dropout(p=self.dropout_probability[1])
        self.fc3 = nn.Linear(self.hidden_size, self.act_dim)

        if self.continuous_actions:
            self.std_fc = nn.Linear(self.hidden_size, self.act_dim)

    def _get_conv_output_size(self, shape):
        # 创建一个假的输入，通过 CNN 层，计算输出的大小
        with paddle.no_grad():
            input = paddle.zeros([1] + list(shape), dtype='float32')
            x = self.conv1(input)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = F.relu(x)
            x = self.pool3(x)
            output_size = x.numel().item()  # 将 Tensor 转换为 Python int
        return output_size

    def forward(self, obs):
        x = self.conv1(obs)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = x.reshape([x.shape[0], -1])  # 展平
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.sigmoid(self.fc2(x))
        x = self.drop2(x)
        means = self.fc3(x)

        if self.continuous_actions:
            means = F.tanh(means)  # 确保输出在 [-1, 1] 范围内
            act_std = self.std_fc(x)
            return means, act_std
        return means


class CriticModel(parl.Model):
    def __init__(self, obs_shape_n, act_shape_n, continuous_actions=False):
        super(CriticModel, self).__init__()
        self.continuous_actions = continuous_actions
        self.n = len(obs_shape_n)  # 智能体数量

        # 定义 CNN 超参数（与 ActorModel 相同）
        self.filter_size = 5
        self.num_filters = [32, 64, 64]
        self.stride = 1
        self.pool_size = 2
        self.hidden_size = 128

        # 定义共享的 CNN 层
        self.conv1 = nn.Conv2D(in_channels=obs_shape_n[0][0], out_channels=self.num_filters[0],
                               kernel_size=self.filter_size, stride=self.stride, padding=2)
        self.pool1 = nn.MaxPool2D(kernel_size=self.pool_size, stride=self.pool_size)

        self.conv2 = nn.Conv2D(in_channels=self.num_filters[0], out_channels=self.num_filters[1],
                               kernel_size=self.filter_size, stride=self.stride, padding=2)
        self.pool2 = nn.MaxPool2D(kernel_size=self.pool_size, stride=self.pool_size)

        self.conv3 = nn.Conv2D(in_channels=self.num_filters[1], out_channels=self.num_filters[2],
                               kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2D(kernel_size=self.pool_size, stride=self.pool_size)

        # 计算单个智能体的特征向量大小
        self.fc_input_size = self._get_conv_output_size(obs_shape_n[0])
        assert isinstance(self.fc_input_size, int), "fc_input_size 应该是一个整数"

        # 计算 Critic 网络的总输入大小
        # 由于观测相同，所以特征提取只做一次，后面只与每个智能体的动作拼接
        self.total_feature_size = self.n * (self.fc_input_size + act_shape_n[0])

        # 定义全连接层
        self.fc1 = nn.Linear(self.total_feature_size, self.hidden_size * 2)
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)

    def _get_conv_output_size(self, shape):
        # 创建一个假的输入，通过 CNN 层，计算输出的大小
        with paddle.no_grad():
            input = paddle.zeros([1] + list(shape), dtype='float32')
            x = self.conv1(input)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = F.relu(x)
            x = self.pool3(x)
            output_size = x.numel().item()  # 将 Tensor 转换为 Python int
        return output_size

    def forward(self, obs_n, act_n):
        """
        前向传播，计算 Q 值。使用一个共享的观测值来减少 CNN 的计算。
        
        Args:
            obs_n (list of paddle.Tensor): 包含所有智能体观测的列表，每个元素形状为 [batch_size, C, H, W]。
            act_n (list of paddle.Tensor): 包含所有智能体动作的列表，每个元素形状为 [batch_size, act_dim]。
        
        Returns:
            paddle.Tensor: Q 值，形状为 [batch_size, 1]。
        """
        # 使用第一个智能体的观测来计算 CNN 特征
        obs = obs_n[0]  # 假设所有智能体的观测相同，使用第一个即可
        x = self.conv1(obs)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        feature = x.reshape([x.shape[0], -1])  # 展平

        # 为每个智能体的动作拼接同样的 CNN 特征
        features = []
        for i in range(self.n):
            act = act_n[i]
            combined_feature = paddle.concat([feature, act], axis=1)
            features.append(combined_feature)

        # 拼接所有智能体的特征和动作
        combined = paddle.concat(features, axis=1)

        # 通过全连接层
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
