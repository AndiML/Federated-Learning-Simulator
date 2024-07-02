"""Represents a module that contains the central learning command."""

import logging
import os

from datetime import datetime
from argparse import Namespace

from federated_learning.commands.base import BaseCommand
from federated_learning.src.saving import ModelSaver
from federated_learning.src.experiments import ExperimentLogger

from federated_learning.src.datasets.data_generator import create_dataset
from federated_learning.src.nn.model_generator import create_global_model
from federated_learning.src.baseline_main import train_baseline_model


class CentralLearningCommand(BaseCommand):
    """Represents a command that represents the central learning algorithm."""

    def __init__(self) -> None:
        """Initializes a new CentralLearning instance."""

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """
        # Selects device for training
        device = 'cuda' if command_line_arguments.use_gpu else 'cpu'
        all_average_train_loss = []
        all_average_train_accuracy = []
        all_validation_loss = []
        all_validation_accuracy = []

        # Creates the ModelSaver for saving the global model in the federated learning training process
        model_directory = os.path.join(command_line_arguments.output_path, "model-checkpoint")
        os.makedirs(model_directory, exist_ok=True)
        self.logger.info('Initializing Model Saver', extra={'start_section': True})
        model_saver = ModelSaver(model_directory=model_directory)

        # Creates the experiment logger for training
        self.logger.info('Initializing Experiment Logger', extra={'start_section': True})
        experiment_logger = ExperimentLogger(command_line_arguments.output_path, baseline=True)
        experiment_logger.display_experiment_hyperparameters_baseline(command_line_arguments, self.logger)

        # Save all hyperparameters utilized in the federated learning process
        hyperparameters = vars(command_line_arguments)
        hyperparameters['start_date_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        experiment_logger.save_hyperparameters(vars(command_line_arguments))

        self.logger.info('Selected %s for Federated Learning Baseline Training Process', device.upper())
        self.logger.info('Retrieve Training and Validation Data', extra={'start_section': True})

        # Loads the datasets for training
        data_class_instance = create_dataset(
                dataset_path=command_line_arguments.dataset_path,
                number_of_clients=1,
                dataset_kind=command_line_arguments.dataset,
                partition_strategy='homogeneous',
                beta=0.0
        )

        # Builds the model
        global_model = create_global_model(
            model_type=command_line_arguments.model_type,
            dataset_kind=command_line_arguments.dataset,
            data_class_instance=data_class_instance,
            tensor_shape_for_flattening=data_class_instance.sample_shape
        )

        # Send the model to the device.
        global_model.to(device)
        current_learning_rate = command_line_arguments.learning_rate
        for epoch in range(command_line_arguments.number_of_epochs):

            # Performs training of the baseline model
            self.logger.info('Starting Training of Model', extra={'start_section': True, 'end_section': True})
            training_loss_per_epoch, training_accuracy_per_epoch, validation_loss, validation_accuracy = train_baseline_model(
                data_class_instance=data_class_instance,
                model=global_model,
                current_epoch=epoch,
                optimizer_kind=command_line_arguments.optimizer,
                batch_size=command_line_arguments.batchsize,
                learning_rate=current_learning_rate,
                momentum=command_line_arguments.set_momentum,
                weight_decay=command_line_arguments.weight_decay,
                loss_function_kind=command_line_arguments.loss_function,
                device=device,
                logger=self.logger,
                experiment_logger=experiment_logger
            )
            # Decays the learning rate
            current_learning_rate *= command_line_arguments.learning_rate_decay

            # Store the results of the current run
            all_average_train_loss.append(training_loss_per_epoch)
            all_average_train_accuracy.append(training_accuracy_per_epoch)
            all_validation_loss.append(validation_loss)
            all_validation_accuracy.append(validation_accuracy)
            if max(all_validation_accuracy) == validation_accuracy:
                model_saver.save_global_model(
                    current_training_round=epoch + 1, baseline=True,
                    global_model_state_dict=global_model.state_dict()
                )

        self.logger.info('Saved all epochs into CSV File')
        # Closes the experiment logger
        if experiment_logger is not None:
            self.logger.info('Closes the experiment logger', extra={'start_section': True})
            experiment_logger.close()
