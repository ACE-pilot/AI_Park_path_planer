import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class MAModel(parl.Model):
    def __init__(self,
                 obs_dim,          # 单个智能体的观测空间形状，例如 (3, 128, 128)
                 act_dim,          # 单个智能体的动作空间维度，例如 2
                 critic_in_dim,    # Critic 网络的输入维度，例如 sum(obs_size_n) + sum(act_dim_n)
                 continuous_actions=False):
        super(MAModel, self).__init__()
        self.actor_model = ActorModel(obs_dim, act_dim, continuous_actions)
        self.critic_model = CriticModel(critic_in_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs_act_flat):
        return self.critic_model(obs_act_flat)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, obs_shape, act_dim, continuous_actions=False):
        super(ActorModel, self).__init__()
        self.continuous_actions = continuous_actions
        print(f"-----ActorModel obs_shape------: {obs_shape}")

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
        print(f"ActorModel fc_input_size: {self.fc_input_size}, type: {type(self.fc_input_size)}")
        assert isinstance(self.fc_input_size, int), "fc_input_size 应该是一个整数"

        self.fc1 = nn.Linear(self.fc_input_size, self.hidden_size * 3)
        self.drop1 = nn.Dropout(p=self.dropout_probability[0])
        self.fc2 = nn.Linear(self.hidden_size * 3, self.hidden_size * 3)
        self.drop2 = nn.Dropout(p=self.dropout_probability[1])
        self.fc3 = nn.Linear(self.hidden_size * 3, self.act_dim)

        if self.continuous_actions:
            self.std_fc = nn.Linear(self.hidden_size * 3, self.act_dim)

    def _get_conv_output_size(self, shape):
        # 创建一个假的输入，通过 CNN 层，计算输出的大小
        with paddle.no_grad():
            input = paddle.zeros([1] + list(shape), dtype='float32')
            x = self.conv1(input)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.pool3(x)
            output_size = x.numel().numpy().item()  # 将 Tensor 转换为 Python int
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
            means = F.tanh(means)  # 确保输出在 [-1, 1] 范围内
            act_std = self.std_fc(x)
            #print(f"ActorModel Action means: {means}")
            #print(f"ActorModel Action std: {act_std}")
            return means, act_std
        #print(f"ActorModel Action means: {means}")
        return means


class CriticModel(parl.Model):
    def __init__(self, critic_in_dim):
        super(CriticModel, self).__init__()
        self.hidden_size = 128

        # 定义全连接层
        self.fc1 = nn.Linear(critic_in_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)

    def forward(self, obs_act_flat):
        x = F.relu(self.fc1(obs_act_flat))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

