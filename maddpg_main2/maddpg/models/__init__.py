from .hyper_model import HyperMAModel
from .mlp_model import MLPMAModel
from .unet_model import UNetMAModel
from .attention_model import AttentionMAModel

MODEL_REGISTRY = {
    'hyper': HyperMAModel,
    'mlp': MLPMAModel,
    'unet': UNetMAModel,
    'attention': AttentionMAModel,
}

# Backward compat: default MAModel is the CNN-MLP (hyper) variant
MAModel = HyperMAModel


def get_model(name: str):
    """Look up a model class by name. Raises KeyError if not found."""
    return MODEL_REGISTRY[name]


__all__ = [
    'MODEL_REGISTRY', 'get_model', 'MAModel',
    'HyperMAModel', 'MLPMAModel', 'UNetMAModel', 'AttentionMAModel',
]
