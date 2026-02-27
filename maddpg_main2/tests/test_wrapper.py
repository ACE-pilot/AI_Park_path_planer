"""Test MADDPGWrapper: n, obs_shape_n, act_shape_n, reset, step."""
import numpy as np
import pytest


@pytest.fixture
def wrapped_env():
    from maddpg.envs import ParkEnv, MADDPGWrapper
    env = ParkEnv(num_agents=3, render_mode=False)
    return MADDPGWrapper(env, continuous_actions=True)


class TestMADDPGWrapper:
    def test_n_agents(self, wrapped_env):
        assert wrapped_env.n == 3

    def test_obs_shape_n_length(self, wrapped_env):
        assert len(wrapped_env.obs_shape_n) == 3

    def test_act_shape_n_length(self, wrapped_env):
        assert len(wrapped_env.act_shape_n) == 3

    def test_obs_shape_is_180(self, wrapped_env):
        for shape in wrapped_env.obs_shape_n:
            assert shape == (180,)

    def test_act_shape_is_4(self, wrapped_env):
        for act_dim in wrapped_env.act_shape_n:
            assert act_dim == 4

    def test_reset_returns_list(self, wrapped_env):
        obs_n = wrapped_env.reset()
        assert isinstance(obs_n, list)
        assert len(obs_n) == 3

    def test_reset_obs_shape(self, wrapped_env):
        obs_n = wrapped_env.reset()
        for obs in obs_n:
            assert obs.shape == (180,)

    def test_step_returns_four_lists(self, wrapped_env):
        wrapped_env.reset()
        actions = [np.random.rand(4).astype(np.float32) for _ in range(3)]
        result = wrapped_env.step(actions)
        assert len(result) == 4
        obs_n, reward_n, done_n, info_n = result
        assert isinstance(obs_n, list) and len(obs_n) == 3
        assert isinstance(reward_n, list) and len(reward_n) == 3
        assert isinstance(done_n, list) and len(done_n) == 3
        assert isinstance(info_n, list) and len(info_n) == 3
