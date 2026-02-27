"""Smoke tests: verify all public imports resolve correctly."""


def test_import_envs():
    from maddpg.envs import ParkEnv, MADDPGWrapper
    assert ParkEnv is not None
    assert MADDPGWrapper is not None


def test_import_models_registry():
    from maddpg.models import get_model, MODEL_REGISTRY
    assert callable(get_model)
    assert isinstance(MODEL_REGISTRY, dict)


def test_import_hyper_model_backward_compat():
    from maddpg.models.hyper_model import MAModel
    assert MAModel is not None


def test_import_mlp_model_backward_compat():
    from maddpg.models.mlp_model import MAModel
    assert MAModel is not None


def test_import_unet_model_backward_compat():
    from maddpg.models.unet_model import MAModel
    assert MAModel is not None


def test_import_attention_model_backward_compat():
    from maddpg.models.attention_model import MAModel
    assert MAModel is not None


def test_import_agents():
    from maddpg.agents import MAAgent
    assert MAAgent is not None


def test_import_viz():
    from maddpg.viz import nature_style
    assert hasattr(nature_style, 'apply')
    assert hasattr(nature_style, 'SCALE_FACTOR')


def test_import_eval_utils():
    from maddpg.eval_utils import setup_env, build_agents, restore_agents
    assert callable(setup_env)
    assert callable(build_agents)
    assert callable(restore_agents)
