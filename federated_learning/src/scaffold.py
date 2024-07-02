"""A module, which contains the federated averaging algorithm."""

import copy
import csv
import logging
import os
from typing import Any

import numpy
import pandas
import torch

from federated_learning.src.experiments import ExperimentLogger
from federated_learning.src.central_server_scaffold import FederatedLearningCentralServerScaffold
from federated_learning.src.saving import ModelSaver


def perform_scaffold(
    central_server: FederatedLearningCentralServerScaffold,
    eval_every: int,
    number_of_clients: int,
    fraction_of_clients: int,
    number_of_training_rounds: int,
    logger: logging.Logger,
    experiment_logger: ExperimentLogger,
    model_saver: ModelSaver,
    run_directory: str

) -> tuple[list[float], list[float]]:
    """ Performs the averaging of the local models retrieved from the participating clients

        Args:
            central_server (FederatedLearningCentralServerScaffold): The central server which updates and validates the
                locally trained models of the clients.
            eval_every (int): The number of times the client performance should be evaluated.
            number_of_clients (int): Number of users that are available in the federated training process.
            fraction_of_clients (int): Subset of users that participate in the federated training process.
            number_of_training_rounds(int): The number of training rounds that the federated training process is carried out.
            logger(logging.Logger): Logger is provided to log metrics obtained during the federated learning process directly to the command line.
            experiment_logger (ExperimentLogger): A logger to save the experiments results and write it to the TensorBoard
            model_saver (ModelSaver): The instance of the ModelSaver class, that saves the parameter of the current global model.
            run_directory(str): The current directory in which the clients performance for the best model is saved.
        Returns:
            tuple[list[float], list[float]]: The validation loss and accuracy of the global model for each communication
                round.
        """

    central_server_accuracy: list[float] = []
    central_server_loss: list[float] = []
    clients_list = central_server.clients

    # Sets up the file for saving the individual client performance
    file_path = os.path.join(run_directory, "client_performance_statistics.csv")
    df = pandas.DataFrame(
        columns=["Communication Round",
                 "Average Client Performance Loss",
                 "Standard Deviation of Performance Loss",
                 "Average Client Performance Accuracy",
                 "Standard Deviation of Performance Accuracy"]
                )
    df.to_csv(file_path, index=False)

    # Notifies the experiment logger that a new epoch has begun
    experiment_logger.begin_training_round(0)

    # Logs the learning rate at the end of each epoch, because the learning rate is decayed and therefore changes each epoch
    experiment_logger.add_metric('learning rate', 'Training/Learning Rate', central_server.get_current_learning_rate())

    central_server_loss_per_communication_round, central_server_accuracy_per_communication_round = central_server.validate_global_model()
    central_server_loss.append(central_server_loss_per_communication_round)
    central_server_accuracy.append(central_server_accuracy_per_communication_round)

    # Logs average the global loss and accuracy for one round of communication
    logger.info(
        'Global Loss: %s, Global Best Loss: %s',
        f'{central_server_loss[-1]:.5f}',
        f'{numpy.min(central_server_loss):.5f}',
        extra={'start_section': True})
    experiment_logger.add_metric('Global Val Loss', 'Global Validation Loss', central_server_loss[-1])
    logger.info(
        'Global Accuracy: %s, Global Best Accuracy: %s',
        f'{central_server_accuracy[-1]:.4f}',
        f'{ numpy.max(central_server_accuracy):.4f}',
        extra={'start_section': True, 'end_section': True})
    experiment_logger.add_metric('Global Val Accuracy', 'Global Validation Accuracy', central_server_accuracy[-1])
    # Notifies the experiment logger that the epoch has finished, which flushes out everything to disk
    experiment_logger.end_training_round()

    # Stars the federated training
    for communication_round in range(number_of_training_rounds):

        global_model_state_dict = central_server.get_global_model_state_dict()
        total_delta = {key: torch.zeros_like(value) for key, value in global_model_state_dict.items()}

        # Notifies the experiment logger that a new epoch has begun
        experiment_logger.begin_training_round(communication_round + 1)

        # Logs the learning rate at the end of each epoch, because the learning rate is decayed and therefore changes each epoch
        experiment_logger.add_metric('learning rate', 'Training/Learning Rate', central_server.get_current_learning_rate())
        # Determines the number of clients participating in the training
        number_of_clients_per_communication_round = max(int(fraction_of_clients * number_of_clients), 1)

        sampled_client_indices = numpy.random.choice(range(number_of_clients), number_of_clients_per_communication_round, replace=False)

        # Starts the training
        logger.info(f'| Starting Global Training Round : {communication_round+1} |', extra={'end_section': True})
        for client_index in sampled_client_indices:
            logger.info(f'| Training Client : {clients_list[client_index].get_client_id() + 1 } |')
            clients_list[client_index].set_state_dict(central_server.get_global_model_state_dict())
            clients_list[client_index].set_learning_rate(central_server.get_current_learning_rate())
            _, _, c_delta_parameter = clients_list[client_index].train(
                copy.deepcopy(central_server.global_control_variate),
                communication_round
            )
            for key in total_delta:
                total_delta[key] += c_delta_parameter[key]

        # Updates the global control variate
        for key in total_delta:
            total_delta[key] /= number_of_clients
        c_global_parameter = central_server.global_control_variate.state_dict()
        for key in c_global_parameter:
            c_global_parameter[key] += total_delta[key]
        central_server.global_control_variate.load_state_dict(c_global_parameter)

        # Updates central server model and validates the central server model

        central_server.update_global_model(sampled_client_indices.tolist())
        # if (communication_round +1)  % eval_every ==0:
        central_server_loss_per_communication_round, central_server_accuracy_per_communication_round = central_server.validate_global_model()
        central_server_loss.append(central_server_loss_per_communication_round)
        central_server_accuracy.append(central_server_accuracy_per_communication_round)

        # Logs average the global loss and accuracy for one round of communication
        logger.info(
            'Global Loss: %s, Global Best Loss: %s',
            f'{central_server_loss[-1]:.5f}',
            f'{numpy.min(central_server_loss):.5f}',
            extra={'start_section': True})
        experiment_logger.add_metric('Global Val Loss', 'Global Validation Loss', central_server_loss[-1])
        logger.info(
            'Global Accuracy: %s, Global Best Accuracy: %s',
            f'{central_server_accuracy[-1]:.4f}',
            f'{ numpy.max(central_server_accuracy):.4f}',
            extra={'start_section': True, 'end_section': True})
        experiment_logger.add_metric('Global Val Accuracy', 'Global Validation Accuracy', central_server_accuracy[-1])

        # Saves global model
        if max(central_server_accuracy) == central_server_accuracy_per_communication_round:
            logger.info("Saving global model after round %d", communication_round + 1,  extra={'end_section': True})
            current_best_model_state_dict = central_server.get_global_model_state_dict()
            model_saver.save_global_model(
                current_training_round=communication_round + 1,
                global_model_state_dict=current_best_model_state_dict)

        # Evaluates the performance of the last updated global model for each client
        if (communication_round + 1) % eval_every == 0:

            client_index_to_accuracy_on_best_model = []
            client_index_to_loss_on_best_model = []
            for client_index in range(number_of_clients):
                logger.info(f'| Evaluate Model Performance of: {clients_list[client_index].get_client_id() + 1} |', extra={'start_section': True})
                clients_list[client_index].set_state_dict(copy.deepcopy(central_server.get_global_model_state_dict()))
                accuracy_client, loss_client = clients_list[client_index].compute_accuracy_and_loss_on_updated_global_model()
                client_index_to_accuracy_on_best_model.append(accuracy_client)
                client_index_to_loss_on_best_model.append(loss_client)

            # Computes statistics based on the average performance
            average_client_training_performance_on_best_model = numpy.mean(client_index_to_accuracy_on_best_model)
            standard_deviation_client_training_performance_on_best_model = numpy.std(client_index_to_accuracy_on_best_model)
            average_client_loss_performance_on_best_model = numpy.mean(client_index_to_loss_on_best_model)
            standard_deviation_client_training_loss_performance_on_best_model = numpy.std(client_index_to_loss_on_best_model)
            new_data = {
                    "Communication Round": [communication_round + 1],
                    "Average Client Performance Loss": [average_client_loss_performance_on_best_model],
                    "Standard Deviation of Performance Loss": [standard_deviation_client_training_loss_performance_on_best_model],
                    "Average Client Performance Accuracy": [average_client_training_performance_on_best_model],
                    "Standard Deviation of Performance Accuracy": [standard_deviation_client_training_performance_on_best_model]
            }
            df_new_round = pandas.DataFrame(new_data)

            # Append new data to the CSV file
            df_new_round.to_csv(file_path, mode='a', header=False, index=False)
            # Add metric to the global model experience logger
            logger.info(f'Performance Clients: \n {client_index_to_accuracy_on_best_model}', extra={'start_section': True})
            logger.info(f'Average Clients Performance Accuracy: {average_client_training_performance_on_best_model}',
                        extra={'start_section': True})
            logger.info(f'Standard deviation Clients Performance Accuracy {standard_deviation_client_training_performance_on_best_model}',
                        extra={'start_section': True})
            logger.info(f'Average Clients Performance Loss: {average_client_loss_performance_on_best_model}',
                        extra={'start_section': True})
            logger.info(f'Standard deviation Clients Performance Loss {standard_deviation_client_training_loss_performance_on_best_model}',
                        extra={'start_section': True})

        # Notifies the experiment logger that the epoch has finished, which flushes out everything to disk
        experiment_logger.end_training_round()

    return central_server_loss, central_server_accuracy
