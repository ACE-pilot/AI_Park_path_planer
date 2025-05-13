import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MAModel(parl.Model):
    def __init__(self, obs_shape, act_dim, critic_obs_shape_n, critic_act_dim_n, continuous_actions=False):
        super(MAModel, self).__init__()
        self.actor_model = ActorModel(obs_shape, act_dim, continuous_actions)
        self.critic_model = CriticModel(critic_obs_shape_n, critic_act_dim_n)

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

        # 定义 CNN 超参数
        self.filter_size = 5
        self.num_filters = [16, 32, 64]
        self.stride = 2
        self.pool_size = 2
        self.dropout_probability = [0.3, 0.2]
        self.hidden_size = 128
        self.act_dim = act_dim

        # 定义 CNN 层
        self.conv1 = nn.Conv2D(in_channels=obs_shape[0], out_channels=self.num_filters[0],
                               kernel_size=self.filter_size, stride=self.stride)
        self.pool1 = nn.MaxPool2D(kernel_size=self.pool_size, stride=self.stride)

        self.conv2 = nn.Conv2D(in_channels=self.num_filters[0], out_channels=self.num_filters[1],
                               kernel_size=self.filter_size, stride=self.stride)
        self.pool2 = nn.MaxPool2D(kernel_size=self.pool_size, stride=self.stride)

        self.conv3 = nn.Conv2D(in_channels=self.num_filters[1], out_channels=self.num_filters[2],
                               kernel_size=self.filter_size, stride=self.stride)
        self.pool3 = nn.MaxPool2D(kernel_size=self.pool_size, stride=self.stride)

        # 计算全连接层的输入大小
        self.fc_input_size = self._get_conv_output_size(obs_shape)

        self.fc1 = nn.Linear(self.fc_input_size, self.hidden_size * 3)
        self.drop1 = nn.Dropout(p=self.dropout_probability[0])
        self.fc2 = nn.Linear(self.hidden_size * 3, self.hidden_size * 3)
        self.drop2 = nn.Dropout(p=self.dropout_probability[1])
        self.fc3 = nn.Linear(self.hidden_size * 3, self.act_dim)

        if self.continuous_actions:
            self.std_fc = nn.Linear(self.hidden_size * 3, self.act_dim)

    def _get_conv_output_size(self, shape):
        # 创建一个假的输入，通过 CNN 层，计算输出的大小
        '''
        假设输入图像的形状为 (3, 64, 64)，即 3 个通道，尺寸为 64x64 的图像。
        我们可以通过这个函数计算出经过 CNN 层后的输出大小：
        output_size = self._get_conv_output_size((3, 64, 64))
        计算过程：
        创建虚拟输入张量，形状为 (1, 3, 64, 64)。
        依次通过卷积和池化层，得到最终特征图 x。
        计算 x.numel()，假设得到 output_size = 1024。
        使用这个输出大小来定义全连接层：
        self.fc1 = nn.Linear(1024, self.hidden_size * 3)
        '''
        with paddle.no_grad():
            input = paddle.zeros([1] + list(shape))
            x = self.conv1(input)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.pool3(x)
            output_size = x.numel()
        return output_size

    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.reshape([x.shape[0], -1])
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.sigmoid(self.fc2(x))
        x = self.drop2(x)
        means = self.fc3(x)
        if self.continuous_actions:
            act_std = self.std_fc(x)
            return means, act_std
        return means


class CriticModel(parl.Model):
    def __init__(self, obs_shape_n, act_dim_n):
        super(CriticModel, self).__init__()
        # 假设所有智能体的观测形状相同
        self.num_agents = len(obs_shape_n)
        self.obs_shape = obs_shape_n[0]

        # 定义 CNN 超参数
        self.filter_size = 5
        self.num_filters = [16, 32, 64]
        self.stride = 2
        self.pool_size = 2
        self.dropout_probability = [0.3, 0.2]
        self.hidden_size = 128

        # 定义 CNN 层（共享）
        self.conv1 = nn.Conv2D(in_channels=self.obs_shape[0], out_channels=self.num_filters[0],
                               kernel_size=self.filter_size, stride=self.stride)
        self.pool1 = nn.MaxPool2D(kernel_size=self.pool_size, stride=self.stride)

        self.conv2 = nn.Conv2D(in_channels=self.num_filters[0], out_channels=self.num_filters[1],
                               kernel_size=self.filter_size, stride=self.stride)
        self.pool2 = nn.MaxPool2D(kernel_size=self.pool_size, stride=self.stride)

        self.conv3 = nn.Conv2D(in_channels=self.num_filters[1], out_channels=self.num_filters[2],
                               kernel_size=self.filter_size, stride=self.stride)
        self.pool3 = nn.MaxPool2D(kernel_size=self.pool_size, stride=self.stride)

        # 计算全连接层的输入大小
        self.fc_obs_input_size = self._get_conv_output_size(self.obs_shape) * self.num_agents
        self.act_dim_total = sum(act_dim_n)
        self.fc_act = nn.Linear(self.act_dim_total, self.hidden_size * 3)

        self.fc1 = nn.Linear(self.fc_obs_input_size + self.hidden_size * 3, self.hidden_size * 3)
        self.drop1 = nn.Dropout(p=self.dropout_probability[0])
        self.fc2 = nn.Linear(self.hidden_size * 3, self.hidden_size * 3)
        self.drop2 = nn.Dropout(p=self.dropout_probability[1])
        self.fc3 = nn.Linear(self.hidden_size * 3, 1)

    def _get_conv_output_size(self, shape):
        # 创建一个假的输入，通过 CNN 层，计算输出的大小
        with paddle.no_grad():
            input = paddle.zeros([1] + list(shape))
            x = self.conv1(input)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.pool3(x)
            output_size = x.numel()
        return output_size

    def forward(self, obs_n, act_n):
        # 处理观测
        obs_processed = []
        for obs in obs_n:
            x = F.relu(self.conv1(obs))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = x.reshape([x.shape[0], -1])
            obs_processed.append(x)
        obs_combined = paddle.concat(obs_processed, axis=1)

        # 处理动作
        act_combined = paddle.concat(act_n, axis=1)
        act_processed = F.relu(self.fc_act(act_combined))

        # 合并观测和动作
        inputs = paddle.concat([obs_combined, act_processed], axis=1)

        x = F.relu(self.fc1(inputs))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        Q = self.fc3(x)
        Q = paddle.squeeze(Q, axis=1)
        return Q
