"""A module, which contains the central server for FedNova."""
from typing import Literal, Any

import torch
from tqdm import tqdm

from federated_learning.src.client_fednova import ClientFedNova


class FederatedLearningFedNovaCentralServer(object):
    """Represents a federated learning central server, which coordinates the federated learning of the global model."""

    def __init__(
        self,
        clients: list[ClientFedNova],
        device: str | torch.device,
        global_model: torch.nn.Module,
        central_server_validation_data_loader: torch.utils.data.DataLoader,
        initial_learning_rate: float,
        loss_function_kind: Literal['mse', 'nll'],
        learning_rate_decay: float,
        batch_size: int = 10
    ) -> None:
        """Initializes a new FederatedLearningCentralServer instance.

        Args:
            clients (list[ClientFedNova]): The list of federated learning clients.
            device (str | torch.device): The device on which the global model of the central server is to be validated.
            global_model (torch.nn.Module): The type of model that is to be used as global model for the central server.
            central_server_validation_data_loader (torch.utils.data.DataLoader): The validation data loader on which the global model is to be
                validated.
            initial_learning_rate (float, optional): The initial learning rate of the optimizer.
            loss_function_kind (Literal['mse', 'nll']): The kind of loss function that is to be used for the training process.
            learning_rate_decay (float, optional): The learning rate is decayed exponentially during the training. This parameter is the decay rate of
                the learning rate. A decay rate 1.0 would result in no decay at all. Defaults to 0.95.
            batch_size (int, optional): The size of the mini-batches that are to be used during the validation. Defaults to 10.

        Raises:
            ValueError: If the specified optimizer or loss function are not supported, an exception is raised.
        """

        # Stores the arguments for later use
        self.clients = clients
        self.device = device
        self.global_model = global_model
        self.central_server_validation_data_loader = central_server_validation_data_loader
        self.current_learning_rate = initial_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.batch_size = batch_size
        self.best_performing_global_model = global_model

        # Creates the loss function
        # Retrieves the loss function for training
        self.loss_function: torch.nn.MSELoss | torch.nn.NLLLoss
        if loss_function_kind == 'mse':
            self.loss_function = torch.nn.MSELoss().to(device)
        elif loss_function_kind == 'nll':
            self.loss_function = torch.nn.NLLLoss().to(device)
        else:
            raise ValueError(f'The loss function "{loss_function_kind}" is not supported.')

        # Sets up global model for training
        self.global_model.to(self.device)
        self.global_model.train()

        # Initializes a flag, which is set when the training should be aborted
        self.is_aborting = False

    def get_global_model_state_dict(self) -> dict[str, Any]:
        """ Retrieves the current global model state dict

        Returns:
            dict[str, Any]: Current global model state dict
        """

        return self.global_model.state_dict()

    def update_global_model(
        self,
        sampled_client_indices: list[int],
        client_to_number_of_local_steps: list[float],
        client_to_normalized_global_update: list[dict[str, Any]]
    ) -> None:
        """Updates the global model of the central server by  aggregating the parameters of the local models of the clients.

        Args:
            sampled_client_indices (list[int]): The list of client indices that is utilized to update the global model of the central server.
            client_to_number_of_local_steps (list[float]): Contains the normalization constant for each sampled client.
            client_to_normalized_global_update (list[dict[str, Any]]): Contains the updated global model parameters for each client.
        """
        # Computes the the weighting of the client models where the weighting is adapted to the number of number of datapoints

        number_of_data_points_total = sum([self.clients[client_index].get_count() for client_index in sampled_client_indices])
        weighting_factor_local_data = [self.clients[client_index].get_count() / number_of_data_points_total
                                       for client_index in sampled_client_indices]

        # Performs Fednova update rule for the models of the participating clients that have been trained locally.
        scaling_factor = 0.0
        d_total_round = {key: torch.zeros_like(value) for key, value in self.global_model.state_dict().items()}

        for index, _ in enumerate(sampled_client_indices):
            d_parameter = client_to_normalized_global_update[index]
            for key in d_parameter:
                d_total_round[key] += d_parameter[key] * weighting_factor_local_data[index]
            scaling_factor += client_to_number_of_local_steps[index] * weighting_factor_local_data[index]

        updated_global_model_state_dict = self.global_model.state_dict()
        for key in updated_global_model_state_dict:
            updated_global_model_state_dict[key] -= scaling_factor * d_total_round[key]

        self.global_model.load_state_dict(updated_global_model_state_dict)

        # Decays the learning rate
        self.current_learning_rate *= self.learning_rate_decay

    def get_current_learning_rate(self) -> float:
        """ Retrieves the current learning rate.

        Returns:
            float: Returns the current learning used for training of the clients.
        """

        return self.current_learning_rate

    @torch.no_grad()
    def validate_global_model(self) -> tuple[float, float]:
        """Validates the global model of the central server.

        Returns:
            tuple[float, float]: Returns the validation loss and the validation accuracy of the global model.
        """
        # Pushes global model to device and sets in evaluation mode
        self.global_model.to(self.device)
        self.global_model.eval()

        validation_loss = 0.0
        number_of_data_points_total = 0.0
        number_of_correct_predictions = 0.0

        # Wrap the validation_data_loader with tqdm to create a progress bar
        progress_bar_validation_data_loader = tqdm(
            self.central_server_validation_data_loader,
            total=len(self.central_server_validation_data_loader),
            desc="| Validate Global Model |"
        )

        # Performs inference
        for (images, labels) in progress_bar_validation_data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.global_model(images)
            validation_loss += self.loss_function(outputs, labels).item()
            _, predicted_labels = torch.max(outputs, dim=1)
            predicted_labels = predicted_labels.view(-1)
            number_of_correct_predictions += torch.sum(torch.eq(predicted_labels, labels)).item()
            number_of_data_points_total += len(labels)

        accuracy = 100. * number_of_correct_predictions / number_of_data_points_total
        # Returns the loss and accuracy of the central server model
        return validation_loss/number_of_data_points_total, accuracy

    def save_global_model(self, output_path: str, current_training_round: int) -> None:
        """Saves the current state of the model to a file.

        Args:
            output_path (str): The path to the directory into which the model is to be saved.
            current_training_round (int): The current epoch of the training.
        """

        torch.save({
            'epoch': current_training_round,
            'model': self.global_model.state_dict()
        }, output_path)
