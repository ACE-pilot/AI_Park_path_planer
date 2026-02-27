"""Test eval_utils helper functions."""
import numpy as np
import pytest


class TestGetElevation:
    def test_returns_float(self):
        from maddpg.eval_utils import get_elevation
        # Fake 3-channel 64x64 map
        env_map = np.random.rand(3, 64, 64).astype(np.float64)
        result = get_elevation(env_map, [10.0, 20.0], 64)
        assert isinstance(result, float)

    def test_clamps_position(self):
        from maddpg.eval_utils import get_elevation
        env_map = np.ones((3, 64, 64)) * 42.0
        result = get_elevation(env_map, [-5.0, 200.0], 64)
        assert result == 42.0


class TestComputeSlope:
    def test_same_point_returns_zero(self):
        from maddpg.eval_utils import compute_slope
        env_map = np.random.rand(3, 64, 64)
        result = compute_slope(env_map, [10, 10], [10, 10], 64, cell_size=1.0)
        assert result == 0.0

    def test_positive_slope(self):
        from maddpg.eval_utils import compute_slope
        env_map = np.zeros((3, 64, 64))
        env_map[1, 10, 10] = 0.0
        env_map[1, 10, 20] = 50.0
        result = compute_slope(env_map, [10, 10], [20, 10], 64, cell_size=1.0)
        assert result > 0
