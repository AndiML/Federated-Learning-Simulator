"""Represents a module that contains the vanilla federated averaging command."""

import logging
import os
import copy
from datetime import datetime
from argparse import Namespace
import random

import numpy
import torch
import pandas

from federated_learning.commands.base import BaseCommand
from federated_learning.src.experiments import ExperimentLogger
from federated_learning.src.saving.model_saver import ModelSaver
from federated_learning.src.client_federated_scaffold import ClientScaffold
from federated_learning.src.central_server_scaffold import FederatedLearningCentralServerScaffold
from federated_learning.src.nn.model_generator import create_global_model
from federated_learning.src.datasets.data_generator import create_dataset
from federated_learning.src.scaffold import perform_scaffold


class ScaffoldCommand(BaseCommand):
    """Represents a command that represents the federated averaging with control variates accounting fo the client drift."""

    def __init__(self) -> None:
        """Initializes a new FederatedAveraging instance. """

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """
        # Selects device for training
        device = 'cuda' if command_line_arguments.use_gpu else 'cpu'
        self.logger.info("Selected %s for Federated Learning Training Process", device.upper())
        all_central_server_loss = []
        all_central_server_accuracy = []

        for run in range(command_line_arguments.number_of_runs):
            random.seed(command_line_arguments.random_seed + run)
            torch.manual_seed(command_line_arguments.random_seed + run)
            numpy.random.seed(command_line_arguments.random_seed + run)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(mode=True)  # type: ignore
            run_directory = os.path.join(command_line_arguments.output_path, f"run[{run+1}]")
            os.makedirs(run_directory, exist_ok=True)

            # Creates the ModelSaver for saving the global model in the federated learning training process
            model_directory = os.path.join(run_directory, "model-checkpoint")
            os.makedirs(model_directory, exist_ok=True)
            self.logger.info('Initializing Model Saver', extra={'start_section': True})
            model_saver = ModelSaver(model_directory=model_directory)

            # Creates the experiment logger
            self.logger.info('Initializing Experiment Logger', extra={'start_section': True})
            experiment_logger = ExperimentLogger(run_directory)
            experiment_logger.display_experiment_hyperparameters_federated(command_line_arguments, self.logger)

            # Save all hyperparameters utilized in the federated learning process
            hyperparameters = vars(command_line_arguments)
            hyperparameters['start_date_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            experiment_logger.save_hyperparameters(vars(command_line_arguments))

            self.logger.info('Starting Run %d of Federated Learning Process', run + 1, extra={'start_section': True})
            self.logger.info('Retrieve Clients Training Data and Central Server Validation Dataset', extra={'start_section': True})

            # Loads the datasets for training
            data_class_instance = create_dataset(
                dataset_path=command_line_arguments.dataset_path,
                number_of_clients=command_line_arguments.number_of_clients,
                dataset_kind=command_line_arguments.dataset,
                partition_strategy=command_line_arguments.iidness,
                beta=command_line_arguments.set_dirichlet_distribution_parameter
            )
            # Builds the global model
            global_model = create_global_model(
                model_type=command_line_arguments.model_type,
                dataset_kind=command_line_arguments.dataset,
                data_class_instance=data_class_instance,
                tensor_shape_for_flattening=data_class_instance.sample_shape
            )
            global_model.to(device)

            # Creates the global control variate
            global_control_variate = copy.deepcopy(global_model).to(device)
            for parameter in global_control_variate.parameters():
                parameter.data.zero_()
            # Adds the chosen model to the experiment for visualization in TensorBoard
            experiment_logger.add_model_graph(
                global_model,
                torch.ones(command_line_arguments.global_batchsize, *data_class_instance.sample_shape).to(device)
            )

            self.logger.info('Initializing Client Models with their Training Data', extra={'start_section': True, 'end_section': True})
            clients_list = []
            for client_id in range(command_line_arguments.number_of_clients):
                self.logger.info("Finished generating Training Data for Client %d", client_id + 1)
                clients_list.append(
                    ClientScaffold(
                        client_id=client_id,
                        model=copy.deepcopy(global_model),
                        local_control_variate=copy.deepcopy(global_control_variate),
                        optimizer_kind=command_line_arguments.optimizer,
                        local_batchsize=command_line_arguments.local_batchsize,
                        local_epochs=command_line_arguments.local_epochs,
                        learning_rate=command_line_arguments.learning_rate,
                        weight_decay=command_line_arguments.weight_decay,
                        momentum=command_line_arguments.set_momentum,
                        device=device,
                        loss_function_kind=command_line_arguments.loss_function,
                        data_class_instance=data_class_instance
                    )
                )

            self.logger.info('Initializing Central Server and Global Model', extra={'start_section': True})
            central_server_validation_data_loader = data_class_instance.get_validation_data_loader(batch_size=command_line_arguments.global_batchsize)
            central_server = FederatedLearningCentralServerScaffold(
                clients=clients_list,
                global_control_variate=global_control_variate,
                device=device,
                global_model=global_model,
                central_server_validation_data_loader=central_server_validation_data_loader,
                initial_learning_rate=command_line_arguments.learning_rate,
                loss_function_kind=command_line_arguments.loss_function,
                learning_rate_decay=command_line_arguments.learning_rate_decay,
                batch_size=command_line_arguments.global_batchsize
            )
            self.logger.info(global_model, extra={'start_section': True})

            # Performs federated averaging
            central_server_loss, central_server_accuracy = perform_scaffold(
                central_server=central_server,
                eval_every=command_line_arguments.eval_every_x_round,
                number_of_clients=command_line_arguments.number_of_clients,
                fraction_of_clients=command_line_arguments.fraction_of_clients,
                number_of_training_rounds=command_line_arguments.number_of_training_rounds,
                logger=self.logger,
                experiment_logger=experiment_logger,
                model_saver=model_saver,
                run_directory=run_directory
            )

            # Store the results of the current run
            all_central_server_loss.append(central_server_loss)
            all_central_server_accuracy.append(central_server_accuracy)

        # Average the results from all runs
        all_central_server_loss = numpy.mean(all_central_server_loss, axis=0)
        all_central_server_accuracy = numpy.mean(all_central_server_accuracy, axis=0)

        # Writes average experiment results into a CSV file
        data_frame = pandas.read_csv(os.path.join(run_directory, "metrics.csv"))
        column_names = data_frame.columns
        data_frame[column_names[-2]] = all_central_server_loss
        data_frame[column_names[-1]] = all_central_server_accuracy

        csv_filename_average_experiments_results = os.path.join(command_line_arguments.output_path, "average_results.csv")
        data_frame.to_csv(csv_filename_average_experiments_results, index=False)

        self.logger.info('Saved Average Run into CSV File', extra={'start_section': True})
        # Closes the experiment logger
        if experiment_logger is not None:
            self.logger.info('Closes the experiment logger', extra={'start_section': True})
            experiment_logger.close()
