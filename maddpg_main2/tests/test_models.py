"""Test that all 4 model variants satisfy the MAModel interface contract."""
import pytest
import paddle
import numpy as np


def _make_model(model_cls, default_model_kwargs):
    return model_cls(**default_model_kwargs)


def _model_classes():
    """Import and return all 4 model classes with their registry keys."""
    from maddpg.models.hyper_model import HyperMAModel
    from maddpg.models.mlp_model import MLPMAModel
    from maddpg.models.unet_model import UNetMAModel
    from maddpg.models.attention_model import AttentionMAModel
    return {
        'hyper': HyperMAModel,
        'mlp': MLPMAModel,
        'unet': UNetMAModel,
        'attention': AttentionMAModel,
    }


class TestModelRegistry:
    def test_registry_has_four_keys(self):
        from maddpg.models import MODEL_REGISTRY
        assert set(MODEL_REGISTRY.keys()) == {'hyper', 'mlp', 'unet', 'attention'}

    def test_get_model_returns_class(self):
        from maddpg.models import get_model
        for key in ('hyper', 'mlp', 'unet', 'attention'):
            cls = get_model(key)
            assert cls is not None

    def test_get_model_unknown_raises(self):
        from maddpg.models import get_model
        with pytest.raises((KeyError, ValueError)):
            get_model('nonexistent')


class TestElevationObsSize:
    def test_hyper_elevation_obs_size(self):
        from maddpg.models.hyper_model import ELEVATION_OBS_SIZE
        assert ELEVATION_OBS_SIZE == 11

    def test_unet_elevation_obs_size(self):
        from maddpg.models.unet_model import ELEVATION_OBS_SIZE
        assert ELEVATION_OBS_SIZE == 11

    def test_attention_elevation_obs_size(self):
        from maddpg.models.attention_model import ELEVATION_OBS_SIZE
        assert ELEVATION_OBS_SIZE == 11


class TestModelInterface:
    """Each model must have policy(), value(), get_actor_params(), get_critic_params()."""

    @pytest.fixture(params=['hyper', 'mlp', 'unet', 'attention'])
    def model(self, request, default_model_kwargs):
        classes = _model_classes()
        cls = classes[request.param]
        return _make_model(cls, default_model_kwargs)

    def test_has_policy(self, model):
        assert hasattr(model, 'policy') and callable(model.policy)

    def test_has_value(self, model):
        assert hasattr(model, 'value') and callable(model.value)

    def test_has_get_actor_params(self, model):
        assert hasattr(model, 'get_actor_params') and callable(model.get_actor_params)

    def test_has_get_critic_params(self, model):
        assert hasattr(model, 'get_critic_params') and callable(model.get_critic_params)

    def test_policy_output_shape(self, model, sample_obs_batch):
        obs = paddle.to_tensor(sample_obs_batch)
        result = model.policy(obs)
        if isinstance(result, tuple):
            means, std = result
            assert means.shape == [4, 4]
        else:
            assert result.shape == [4, 4]

    def test_value_output_shape(self, model, sample_obs_batch, sample_act_batch):
        batch_size = sample_obs_batch.shape[0]
        obs_n = [paddle.to_tensor(sample_obs_batch) for _ in range(3)]
        act_n = [paddle.to_tensor(sample_act_batch) for _ in range(3)]
        q = model.value(obs_n, act_n)
        assert q.shape == [batch_size]

    def test_actor_params_not_empty(self, model):
        params = list(model.get_actor_params())
        assert len(params) > 0

    def test_critic_params_not_empty(self, model):
        params = list(model.get_critic_params())
        assert len(params) > 0
