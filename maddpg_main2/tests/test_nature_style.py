"""Test nature_style visualization utilities."""
import numpy as np
import pytest


class TestNatureStyle:
    def test_scale_factor_is_20(self):
        from maddpg.viz.nature_style import SCALE_FACTOR
        assert SCALE_FACTOR == 20

    def test_scale_episodes(self):
        from maddpg.viz.nature_style import scale_episodes
        result = scale_episodes([0, 1000])
        expected = np.array([0, 20000])
        np.testing.assert_array_equal(result, expected)

    def test_smooth_short_array_no_crash(self):
        from maddpg.viz.nature_style import smooth
        data = np.array([1.0, 2.0, 3.0])
        result = smooth(data, window=500)
        assert len(result) > 0

    def test_smooth_normal(self):
        from maddpg.viz.nature_style import smooth
        data = np.ones(1000)
        result = smooth(data, window=10)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_apply_does_not_crash(self):
        from maddpg.viz.nature_style import apply
        apply()  # should not raise

    def test_color_list_has_five(self):
        from maddpg.viz.nature_style import COLOR_LIST
        assert len(COLOR_LIST) == 5
