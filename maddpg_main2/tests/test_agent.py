"""Test MAAgent interface: predict, sample, add_experience."""
import numpy as np
import pytest
import paddle


@pytest.fixture
def agent():
    """Build a single MAAgent using HyperMAModel."""
    from maddpg.models.hyper_model import HyperMAModel
    from maddpg.agents import MAAgent
    from parl.algorithms import MADDPG
    from gym import spaces

    obs_shape_n = [(180,), (180,), (180,)]
    act_shape_n = [4, 4, 4]

    model = HyperMAModel(
        obs_dim=(180,),
        act_dim=4,
        obs_shape_n=obs_shape_n,
        act_shape_n=act_shape_n,
        continuous_actions=True,
    )
    act_space_n = [
        spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        for _ in range(3)
    ]
    alg = MADDPG(
        model,
        agent_index=0,
        act_space=act_space_n,
        gamma=0.95,
        tau=0.001,
        critic_lr=0.001,
        actor_lr=0.0001,
    )
    return MAAgent(
        alg,
        agent_index=0,
        obs_dim_n=obs_shape_n,
        act_dim_n=act_shape_n,
        batch_size=32,
    )


class TestMAAgent:
    def test_predict_returns_numpy(self, agent):
        obs = np.random.randn(180).astype(np.float32)
        act = agent.predict(obs)
        assert isinstance(act, np.ndarray)

    def test_predict_shape(self, agent):
        obs = np.random.randn(180).astype(np.float32)
        act = agent.predict(obs)
        assert act.shape == (4,)

    @pytest.mark.skipif(
        paddle.get_device().startswith('cpu'),
        reason='MADDPG.sample() uses paddle.normal with shape bug on CPU'
    )
    def test_sample_returns_numpy(self, agent):
        obs = np.random.randn(180).astype(np.float32)
        act = agent.sample(obs)
        assert isinstance(act, np.ndarray)

    @pytest.mark.skipif(
        paddle.get_device().startswith('cpu'),
        reason='MADDPG.sample() uses paddle.normal with shape bug on CPU'
    )
    def test_sample_shape(self, agent):
        obs = np.random.randn(180).astype(np.float32)
        act = agent.sample(obs)
        assert act.shape == (4,)

    def test_add_experience_increases_rpm(self, agent):
        obs = np.random.randn(180).astype(np.float32)
        act = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(180).astype(np.float32)

        size_before = agent.rpm.size()
        agent.add_experience(obs, act, 1.0, next_obs, False)
        assert agent.rpm.size() == size_before + 1
