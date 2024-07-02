"""Represents a module containing the creation of the global model for the federated learning process."""
import sys
import torch

from federated_learning.src.nn.mlp import MLP
from federated_learning.src.nn.cnnmnist import CNNMnist
from federated_learning.src.nn.cnnfmnist import CNNFmnist
from federated_learning.src.nn.cnncifar10 import CNNCifar10
from federated_learning.src.nn.cnncifar100 import CNNCifar100
from federated_learning.src.datasets.dataset import Dataset


def create_global_model(
        model_type: str,
        dataset_kind: str,
        data_class_instance: Dataset,
        tensor_shape_for_flattening: tuple[int, ...]
) -> torch.nn.Module:
    """Creates the global model for the federated learning process

    Args:
        model_type (str): The type of model architecture to be used.
        dataset_kind (str): The kind of dataset used in the federated learning process.
        data_class_instance (Dataset): The instance of the class containing the dataset.
        tensor_shape_for_flattening (tuple[int, ...]): The shape of single sample in the training to initialize the first linear layer of
            a simple feed forward neural network.

    Returns:
        torch.nn.Module: The global model for the federated learning process.
    """
    global_model: torch.nn.Module
    if model_type == 'cnn':
        # Convolutional neural network architecture
        if dataset_kind == 'mnist':
            global_model = CNNMnist(number_of_channels=data_class_instance.sample_shape[0], output_classes=data_class_instance.number_of_classes)
        elif dataset_kind == 'fmnist' or dataset_kind == 'femnist' or dataset_kind == 'emnist':
            global_model = CNNFmnist(number_of_channels=data_class_instance.sample_shape[0], output_classes=data_class_instance.number_of_classes)
        elif dataset_kind == 'cifar10' or dataset_kind == 'cifar100-super' or dataset_kind == 'cinic10':
            global_model = CNNCifar10(number_of_channels=data_class_instance.sample_shape[0], output_classes=data_class_instance.number_of_classes)
        elif dataset_kind == 'cifar100':
            global_model = CNNCifar100(number_of_channels=data_class_instance.sample_shape[0], output_classes=data_class_instance.number_of_classes)
        else:
            exit('Model not supported.')

    elif model_type == 'mlp':
        # Multi-layer perceptron
        img_size = tensor_shape_for_flattening
        input_dimension = 1
        for size_per_dimension in img_size:
            input_dimension *= size_per_dimension
        global_model = MLP(input_dimension=input_dimension, output_classes=data_class_instance.number_of_classes)

    else:
        sys.exit('Architecture not supported')

    return global_model
