"""Represents a module that contains experiment management utilities."""

import os
import csv
import logging
from datetime import datetime
from argparse import Namespace
from typing import Any

import yaml  # type: ignore
import torch
from torch.utils.tensorboard.writer import SummaryWriter


class ExperimentLogger:
    """Represents an experiment logger, which collects information about the model, hyperparameters, and logs of a single experiment."""

    def __init__(self, output_path: str, baseline: bool = False) -> None:
        """Initializes a new ExperimentLogger instance.

        Args:
            output_path (str): The path to the directory into which the experiment information is to be written.
            baseline (bool): If the switch is set, the training without federated learning is executed. Therefore the column training round has to be
                replaced by epoch.
        """

        self.output_path = output_path

        self.file_writer = open(os.path.join(self.output_path, 'metrics.csv'), 'w', encoding='utf-8')
        self.csv_writer = csv.writer(self.file_writer)
        self.csv_header = ['Epoch' if baseline else 'Training Round', 'Timestamp']
        self.has_written_csv_header = False

        self.tensorboard_summary_writer = SummaryWriter(log_dir=self.output_path)  # type: ignore
        self.current_training_round: int | None
        self.current_training_round = None
        self.current_training_round_metrics: dict[str, Any]
        self.current_training_round_metrics = {}

    def display_experiment_hyperparameters_federated(self, command_line_arguments: Namespace, logger: logging.Logger) -> None:
        """Displays the parameters that are used during the federated learning process.

        Args:
            command_line_arguments (Namespace): The command line arguments that are used in the federating learning training process.
            logger(logging.Logger): Logger is provided to log metrics obtained during the federated learning process directly to the command line.

        """
        logger.info('\nExperimental details:')
        logger.info(f'Model                             : {command_line_arguments.model_type.upper()}')
        logger.info(f'Optimizer                         : {command_line_arguments.optimizer.upper()}')
        logger.info(f'Learning Rate                     : {command_line_arguments.learning_rate}')
        logger.info(f'Communication Rounds              : {command_line_arguments.number_of_training_rounds}\n')
        logger.info('Federated Parameters')
        logger.info(f'Data Generating Distribution      : {command_line_arguments.iidness}')
        logger.info(f'Fraction of Clients for Training  : {command_line_arguments.fraction_of_clients}')
        logger.info(f'Local Batch size                  : {command_line_arguments.local_batchsize}')
        logger.info(f'Loss function                     : {command_line_arguments.loss_function}')
        logger.info(f'GPU enabled:                      : {command_line_arguments.use_gpu}')
        logger.info(f'Local Epochs                      : {command_line_arguments.local_epochs}\n')

    def display_experiment_hyperparameters_baseline(self, command_line_arguments: Namespace, logger: logging.Logger) -> None:
        """Displays the parameters that are used during the federated learning process.

        Args:
            command_line_arguments (Namespace): The command line arguments that are used in the federating learning training process.
            logger(logging.Logger): Logger is provided to log metrics obtained during the federated learning process directly to the command line.

        """
        logger.info('\nExperimental details:')
        logger.info(f'Model                             : {command_line_arguments.model_type.upper()}')
        logger.info(f'Optimizer                         : {command_line_arguments.optimizer.upper()}')
        logger.info(f'Learning Rate                     : {command_line_arguments.learning_rate}')
        logger.info(f'Epochs                            : {command_line_arguments.number_of_epochs}\n')
        logger.info('Training Parameters')
        logger.info(f'Batch size                        : {command_line_arguments.batchsize}')
        logger.info(f'Loss function                     : {command_line_arguments.loss_function}')
        logger.info(f' GPU enabled:                     : {command_line_arguments.use_gpu}\n')

    def save_hyperparameters(self, hyperparameters: dict[str, int | float | str | bool]) -> None:
        """Saves the specified hyperparameters in a JSON file.

        Args:
            hyperparameters (dict[str, int | float | str |bool]): A dictionary of name-value pairs, which is to be stored.
        """

        # Saves the hyperparameters into a YAML file
        with open(os.path.join(self.output_path, 'hyperparameters.yaml'), 'w', encoding='utf-8') as hyperparameters_file:
            yaml.dump(hyperparameters, hyperparameters_file)

    def begin_training_round(self, training_round: int, baseline: bool = False) -> None:
        """Begins a new training round.

        Args:
            training_round (int): The current training round (this can be an epoch in case of regular training or a communication round in the case of
                federated learning).
            baseline (bool): If the switch is set, the training without federated learning is executed. Therefore the column training round has to be
                replaced by epoch.
        """
        self.current_training_round = training_round
        self.current_training_round_metrics = {
            'Epoch' if baseline else 'Training Round': training_round,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def add_metric(self, name: str, title: str, value: float) -> None:
        """Adds metrics to the log for the current training round.

        Args:
            name (str): The name of the metric. This is used as the header in the resulting CSV file.
            title (str): The human-readable name of the metric. This is used as the title in TensorBoard.
            value (float): The value of the metric.
        """

        # Checks if the metric is new, if so, then it is added to the header (unless the header has already been written to file, in that case the
        # number of columns cannot be changed)
        if not self.has_written_csv_header and name not in self.csv_header:
            self.csv_header.append(name)

        # Writes the metric to the TensorBoard log file
        if value is not None:
            self.tensorboard_summary_writer.add_scalar(title, value, self.current_training_round)  # type: ignore
        # Add another header to log the average performance of the client

        # Caches the metric, it will be written to disk once the metrics are committed
        self.current_training_round_metrics[name] = value

    def add_model_graph(
            self,
            model: torch.nn.Module,
            inputs: torch.Tensor) -> None:
        """Adds the model graph to the TensorBoard logs.

        Args:
            model (torch.nn.Module): The model whose graph is to be added to the TensorBoard log.
            inputs (torch.Tensor): A sample input, which is used to simulate a forward pass through the network, to generate the model graph (PyTorch
                has a dynamic execution graph, which is built up every time the model is invoked).
        """

        self.tensorboard_summary_writer.add_graph(model, inputs)  # type: ignore

    def end_training_round(self) -> None:
        """Ends the current training round and commits the current training round's metrics to the metrics file."""

        # If the header has not, yet, been written (i.e., this is the first time metrics have been committed), then the header is written to file
        if not self.has_written_csv_header:
            self.csv_writer.writerow(self.csv_header)
            self.has_written_csv_header = True

        # Writes the metrics to the metrics file
        metrics = []
        for metric_name in self.csv_header:
            if metric_name in self.current_training_round_metrics:
                metrics.append(self.current_training_round_metrics[metric_name])
            else:
                metrics.append(None)
        self.csv_writer.writerow(metrics)
        self.file_writer.flush()

    def close(self) -> None:
        """Closes the metrics file."""

        if not self.file_writer.closed:
            self.file_writer.close()

        self.tensorboard_summary_writer.close()  # type: ignore
