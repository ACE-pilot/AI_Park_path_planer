import parl
import paddle
import numpy as np
from parl.utils import ReplayMemory


class MAAgent(parl.Agent):
    def __init__(self,
                 algorithm,
                 agent_index=None,
                 obs_dim_n=None,
                 act_dim_n=None,
                 batch_size=None):
        assert isinstance(agent_index, int)
        assert isinstance(obs_dim_n, list)
        assert isinstance(act_dim_n, list)
        assert isinstance(batch_size, int)
        self.agent_index = agent_index
        self.obs_dim_n = obs_dim_n
        self.act_dim_n = act_dim_n
        self.batch_size = batch_size
        self.n = len(act_dim_n)

        self.memory_size = int(1e6)  # change 1e6 to 1e5
        self.min_memory_size = batch_size * 25  # batch_size * args.max_episode_len

        # 计算单个智能体的展平观测维度和动作维度
        obs_shape = self.obs_dim_n[self.agent_index]  # 例如 (3, 128, 128)
        obs_dim = int(np.prod(obs_shape))             # 例如 49152
        act_dim = self.act_dim_n[self.agent_index]    # 例如 2

        # 初始化 ReplayMemory，不传递 batch_size 参数
        self.rpm = ReplayMemory(
            max_size=self.memory_size,
            obs_dim=obs_dim,
            act_dim=act_dim
        )
        self.global_train_step = 0

        super(MAAgent, self).__init__(algorithm)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)

    def predict(self, obs):
        """ predict action by model
        """
        # 将观测数据调整为模型期望的形状
        obs = obs.reshape(1, *self.obs_dim_n[self.agent_index])
        obs = paddle.to_tensor(obs, dtype='float32')
        act = self.alg.predict(obs)
        act_numpy = act.detach().cpu().numpy().flatten()
        return act_numpy

    def sample(self, obs, use_target_model=False):
        """ sample action by model or target_model
        """
        obs = obs.reshape(1, *self.obs_dim_n[self.agent_index])
        obs = paddle.to_tensor(obs, dtype='float32')
        act = self.alg.sample(obs, use_target_model=use_target_model)
        act_numpy = act.detach().cpu().numpy().flatten()
        return act_numpy

    def learn(self, agents):
        """ sample batch, compute q_target and train
        """
        self.global_train_step += 1

        # only update parameter every 100 steps
        if self.global_train_step % 100 != 0:
            return 0.0

        if self.rpm.size() <= self.min_memory_size:
            return 0.0

        batch_obs_n = []
        batch_act_n = []
        batch_obs_next_n = []

        # sample batch
        rpm_sample_index = self.rpm.make_index(self.batch_size)
        for i in range(self.n):
            batch_obs, batch_act, _, batch_obs_next, _ \
                = agents[i].rpm.sample_batch_by_index(rpm_sample_index)
            # 将观测数据重新调整为原始形状
            obs_shape = self.obs_dim_n[i]
            batch_obs = batch_obs.reshape(-1, *obs_shape)
            batch_obs_next = batch_obs_next.reshape(-1, *obs_shape)
            batch_obs_n.append(batch_obs)
            batch_act_n.append(batch_act)
            batch_obs_next_n.append(batch_obs_next)
        _, _, batch_rew, _, batch_isOver = self.rpm.sample_batch_by_index(
            rpm_sample_index)
        batch_rew = paddle.to_tensor(batch_rew, dtype='float32')
        batch_isOver = paddle.to_tensor(batch_isOver, dtype='float32')

        # 将数据转换为 tensor
        batch_obs_n = [
            paddle.to_tensor(obs, dtype='float32') for obs in batch_obs_n
        ]
        batch_act_n = [
            paddle.to_tensor(act, dtype='float32') for act in batch_act_n
        ]
        batch_obs_next_n = [
            paddle.to_tensor(obs, dtype='float32') for obs in batch_obs_next_n
        ]

        # compute target q
        target_act_next_n = []
        for i in range(self.n):
            target_act_next = agents[i].alg.sample(
                batch_obs_next_n[i], use_target_model=True)
            target_act_next = target_act_next.detach()
            target_act_next_n.append(target_act_next)
        target_q_next = self.alg.Q(
            batch_obs_next_n, target_act_next_n, use_target_model=True)
        target_q = batch_rew + self.alg.gamma * (
            1.0 - batch_isOver) * target_q_next.detach()

        # learn
        critic_cost = self.alg.learn(batch_obs_n, batch_act_n, target_q)
        critic_cost = float(critic_cost.cpu().detach())

        return critic_cost

    def add_experience(self, obs, act, reward, next_obs, terminal):
        # 将观测数据展平成一维数组
        obs_flat = obs.flatten()
        next_obs_flat = next_obs.flatten()
        self.rpm.append(obs_flat, act, reward, next_obs_flat, terminal)
