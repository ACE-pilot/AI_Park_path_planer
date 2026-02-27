"""Shared test fixtures for maddpg tests."""
import numpy as np
import pytest

# Force CPU for tests (cudnn may not be available)
import paddle
paddle.set_device('cpu')


@pytest.fixture
def sample_obs_180():
    """Return a single 180-dim observation vector (float32)."""
    return np.random.randn(180).astype(np.float32)


@pytest.fixture
def sample_obs_batch():
    """Return a batch of 4 observations, shape (4, 180)."""
    return np.random.randn(4, 180).astype(np.float32)


@pytest.fixture
def sample_act_batch():
    """Return a batch of 4 actions, shape (4, 4)."""
    return np.random.randn(4, 4).astype(np.float32)


@pytest.fixture
def obs_shape_n():
    """Standard 3-agent observation shape list."""
    return [(180,), (180,), (180,)]


@pytest.fixture
def act_shape_n():
    """Standard 3-agent action dimension list."""
    return [4, 4, 4]


@pytest.fixture
def default_model_kwargs(obs_shape_n, act_shape_n):
    """Default keyword args for constructing any MAModel variant."""
    return dict(
        obs_dim=(180,),
        act_dim=4,
        obs_shape_n=obs_shape_n,
        act_shape_n=act_shape_n,
        continuous_actions=False,
    )
