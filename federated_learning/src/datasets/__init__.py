"""A sub-package that contains datasets and algorithms for partitioning datasets for federated learning."""

from federated_learning.src.datasets.mnist import Mnist
from federated_learning.src.datasets.fmnist import Fmnist
from federated_learning.src.datasets.femnist import Femnist
from federated_learning.src.datasets.emnist import Emnist
from federated_learning.src.datasets.cifar10 import Cifar10
from federated_learning.src.datasets.cinic10 import Cinic10
from federated_learning.src.datasets.cifar100 import Cifar100
from federated_learning.src.datasets.cifar100_super import Cifar100Super
from federated_learning.src.datasets.dataset import Dataset
from federated_learning.src.datasets.data_generator import DatasetSplit, create_dataset

DATASET_IDS = [
    Cifar100.dataset_id,
    Cifar10.dataset_id,
    Cifar100Super.dataset_id,
    Cinic10.dataset_id,
    Fmnist.dataset_id,
    Emnist.dataset_id,
    Femnist.dataset_id,
    Mnist.dataset_id
]
"""Contains the IDs of all available datasets."""

DEFAULT_DATASET_ID = Mnist.dataset_id
"""Contains the ID of the default dataset."""

__all__ = [
    'Mnist',
    'Fmnist',
    'Emnist',
    'Femnist',
    'Cifar10',
    'Cinic10',
    'Cifar100'
    'cifar100-super',
    'Dataset',

    'DatasetSplit',
    'create_dataset',

    'DATASET_IDS',
    'DEFAULT_DATASET_ID'
]
