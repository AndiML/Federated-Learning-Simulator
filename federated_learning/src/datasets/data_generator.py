"""Represents a module containing the creation of the data for the federated learning process."""

from importlib import import_module
from inspect import getmembers, isclass
from typing import Any

from federated_learning.src.datasets.dataset import Dataset, DatasetSplit
from federated_learning.src.datasets.partition import DataPartitioner


def create_dataset(
    dataset_path: str,
    number_of_clients: int,
    dataset_kind: str,
    partition_strategy: str,
    beta: float
) -> Dataset:
    """Creates the datasets for each client according to the specified partition strategy

    Args:
        dataset_path (str): The path were the dataset is stored. If the dataset could not be found, then it is automatically downloaded to the
            specified location.
        number_of_clients (int): Number of clients to partition the dataset.
        dataset_kind (str): The kind of dataset that is used for local training.
        partition_strategy (str): The strategy to split the dataset among the clients.
        beta (float): he parameter of the dirichlet distribution which controls the heterogeneity of the clients' local data.
    Raises:
        ValueError: If the sub-class that implements the abstract base class for datasets did not specify a sample shape, an exception is raised.

    Returns:
        Dataset: A list that contains the training data for each client.
    """
    # Loads the class corresponding to the specified dataset
    dataset_module = import_module("federated_learning.src.datasets")
    dataset_module_classes = getmembers(dataset_module, isclass)
    dataset_class: type | None = None
    for _, class_object in dataset_module_classes:
        if Dataset in class_object.__bases__ and hasattr(class_object, 'dataset_id') and getattr(class_object, 'dataset_id') == dataset_kind:
            dataset_class = class_object
            break
    if dataset_class is None:
        raise ValueError(f'No dataset of the specified kind "{dataset_kind}" could be found.')
    data_class_instance: Dataset = dataset_class(dataset_path)

    # Creates a instance of the DataPartitioner class which implements the specified data splitting strategy
    data_partitioner_instance = DataPartitioner(
        train_dataset_instance=data_class_instance,
        number_of_clients=number_of_clients,
        beta=beta
    )
    # Applies the specified partition strategy
    client_index_to_train_indices = {}

    if data_class_instance.dataset_id == 'femnist':
        client_index_to_train_indices = data_partitioner_instance.partition_real_world()

    elif partition_strategy == 'homogeneous':
        client_index_to_train_indices = data_partitioner_instance.partition_data_homogeneous()

    elif partition_strategy == 'label-imbalance':
        client_index_to_train_indices = data_partitioner_instance.partition_data_based_on_dirichlet_generated_label_imbalances()

    elif partition_strategy.startswith('noniid-label'):
        client_index_to_train_indices = data_partitioner_instance.partition_data_with_fixed_subset_of_labels(int(partition_strategy[12:]))

    elif partition_strategy == 'vary-datasize':
        client_index_to_train_indices = data_partitioner_instance.partition_data_with_varying_sample_size()
    else:
        exit("Partition strategy not supported")

    for client_index in client_index_to_train_indices.keys():

        data_indices_client = client_index_to_train_indices[client_index].tolist()

        # Adds the partitioned training data for the client to the instantiated data class
        training_split_client = DatasetSplit(data_class_instance.training_data, data_indices_client)
        data_class_instance.partitioned_training_data.append(training_split_client)

    # Returns a list that contains the training data for each client and the central server validation dataset
    return data_class_instance

