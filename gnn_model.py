# gnn_model.py
import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from pgl import Graph  # 导入 PGL 的 Graph
from pgl.nn import GCNConv  # 从 pgl.nn 中导入 GCNConv
import numpy as np


class MAModel(parl.Model):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 obs_shape_n,
                 act_shape_n,
                 continuous_actions=False):
        super(MAModel, self).__init__()
        self.actor_model = GNNActorModel(obs_dim, act_dim, continuous_actions)
        self.critic_model = GNNCriticModel(obs_shape_n, act_shape_n, continuous_actions)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs_n, act_n):
        return self.critic_model(obs_n, act_n)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class GNNActorModel(parl.Model):
    def __init__(self, obs_shape, act_dim, continuous_actions=False):
        super(GNNActorModel, self).__init__()
        self.continuous_actions = continuous_actions
        self.hidden_size = 128
        self.act_dim = act_dim

        self.conv1 = GCNConv(input_size=3, output_size=self.hidden_size)
        self.conv2 = GCNConv(input_size=self.hidden_size, output_size=self.hidden_size)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, act_dim)

        if self.continuous_actions:
            self.std_fc = nn.Linear(self.hidden_size, act_dim)

    def forward(self, obs):
        node_features, edge_index, agent_idx = self._preprocess_observations(obs)

        # 创建图结构
        graph = Graph(num_nodes=node_features.shape[0], edges=edge_index)

        # 使用 GNN 提取特征
        x = self.conv1(graph, node_features)
        x = F.relu(x)
        x = self.conv2(graph, x)
        x = F.relu(x)

        # 获取智能体节点的特征
        if agent_idx >= node_features.shape[0]:
            raise ValueError(f"agent_idx {agent_idx} is out of bounds for number of nodes {node_features.shape[0]}")
        agent_feature = x[agent_idx]

        # 映射到动作空间
        x = F.relu(self.fc1(agent_feature))
        means = self.fc2(x)

        if self.continuous_actions:
            means = F.tanh(means)
            act_std = self.std_fc(x)
            return means, act_std
        return means

    def _preprocess_observations(self, obs):
        object_layer = obs[:, 2, :, :].squeeze()
        object_positions = (object_layer > 0).nonzero(as_tuple=False)
        object_types = object_layer[object_positions[:, 0], object_positions[:, 1]].unsqueeze(-1)

        node_features = paddle.concat([object_positions.astype('float32'), object_types.astype('float32')], axis=-1)
        num_nodes = node_features.shape[0]

        if num_nodes == 0:
            raise ValueError("No nodes found in the observation.")

        # 使用 Paddle Tensor 创建 edge_index
        # 注意：如果节点数量过多，完全连接可能导致内存问题。根据需要调整边的生成方式。
        edge_index = paddle.to_tensor(
            [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
            dtype='int64'
        ).T  # Transpose to get shape [2, num_edges]

        agent_idx = 0  # 假设第一个物体是智能体
        return node_features, edge_index, agent_idx


class GNNCriticModel(parl.Model):
    def __init__(self, obs_shape_n, act_shape_n, continuous_actions=False):
        super(GNNCriticModel, self).__init__()
        self.continuous_actions = continuous_actions
        self.hidden_size = 128
        self.n = len(obs_shape_n)

        self.conv1 = GCNConv(input_size=3, output_size=self.hidden_size)
        self.conv2 = GCNConv(input_size=self.hidden_size, output_size=self.hidden_size)

        self.fc1 = nn.Linear(self.hidden_size + sum(act_shape_n), self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)

    def forward(self, obs_n, act_n):
        obs = obs_n[0]
        node_features, edge_index, agent_idx = self._preprocess_observations(obs)

        # 创建图结构
        graph = Graph(num_nodes=node_features.shape[0], edges=edge_index)

        x = self.conv1(graph, node_features)
        x = F.relu(x)
        x = self.conv2(graph, x)
        x = F.relu(x)

        # 获取智能体节点的特征
        if agent_idx >= node_features.shape[0]:
            raise ValueError(f"agent_idx {agent_idx} is out of bounds for number of nodes {node_features.shape[0]}")
        agent_feature = x[agent_idx]
        combined_features = paddle.concat([agent_feature] + act_n, axis=-1)

        x = F.relu(self.fc1(combined_features))
        q_value = self.fc2(x)
        return q_value

    def _preprocess_observations(self, obs):
        object_layer = obs[:, 2, :, :].squeeze()
        object_positions = (object_layer > 0).nonzero(as_tuple=False)
        object_types = object_layer[object_positions[:, 0], object_positions[:, 1]].unsqueeze(-1)

        node_features = paddle.concat([object_positions.astype('float32'), object_types.astype('float32')], axis=-1)
        num_nodes = node_features.shape[0]

        if num_nodes == 0:
            raise ValueError("No nodes found in the observation.")

        # 使用 Paddle Tensor 创建 edge_index
        edge_index = paddle.to_tensor(
            [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
            dtype='int64'
        ).T  # Transpose to get shape [2, num_edges]

        agent_idx = 0  # 假设第一个物体是智能体
        return node_features, edge_index, agent_idx
