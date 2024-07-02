"""Represents a module that contains the vanilla federated averaging command."""

import logging
from argparse import Namespace
from importlib import import_module
from inspect import getmembers, isclass

from federated_learning.commands.base import BaseCommand
from federated_learning.src.datasets import DATASET_IDS
from federated_learning.src.datasets.dataset import Dataset


class DownloadTorchvisionDatasetsCommand(BaseCommand):
    """Represents a command that represents the federated averaging algorithm."""

    def __init__(self) -> None:
        """Initializes a new FederatedAveraging instance. """

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """
        # Downloads the specified dataset
        self.logger.info("Downloading %s Dataset for Federated Learning Training Process", command_line_arguments.dataset.upper())

        if command_line_arguments.dataset in DATASET_IDS:
            # Loads the class corresponding to the specified dataset
            dataset_module = import_module("federated_learning.src.datasets")
            dataset_module_classes = getmembers(dataset_module, isclass)
            for _, class_object in dataset_module_classes:
                if Dataset in class_object.__bases__ and hasattr(class_object, 'dataset_id') \
                        and getattr(class_object, 'dataset_id') == command_line_arguments.dataset:

                    dataset_class = class_object
                    break
            dataset_class.download(command_line_arguments.dataset_path)
        else:
            exit("Dataset not supported")
