"""A sub-package that contains models and algorithms for generating the global model and local model for federated learning."""

from federated_learning.src.nn.mlp import MLP
from federated_learning.src.nn.cnnmnist import CNNMnist
from federated_learning.src.nn.cnnfmnist import CNNFmnist
from federated_learning.src.nn.cnncifar10 import CNNCifar10
from federated_learning.src.nn.model_generator import create_global_model

MODEL_IDS = [MLP.model_id, CNNMnist.model_id]
"""Contains the IDs of all available model architectures."""

DEFAULT_MODEL_ID = MLP.model_id
"""Contains the ID of the default model architecture."""

__all__ = [
    'MLP',
    'CNNMnist',
    'CNNFmnist',
    'CNNCifar10',

    'create_global_model',

    'MODEL_IDS',
    'DEFAULT_MODEL_ID'
]
