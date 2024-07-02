"""A module, which contains the client for SCAFFOLD."""
import copy
from typing import Any, Literal
import torch
from tqdm import tqdm

from federated_learning.src.datasets.dataset import Dataset


class ClientScaffold(object):
    """Represents a client in federated learning setting that receives model parameters from the central server and trains it on its local dataset."""

    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        local_control_variate: torch.nn.Module,
        optimizer_kind: Literal['sgd', 'adam'],
        local_batchsize: int,
        local_epochs: int,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        device: str | torch.device,
        loss_function_kind: Literal['mse', 'nll'],
        data_class_instance: Dataset,
        save_best_model: bool = False
    ) -> None:
        """Initializes a new FederatedLearningClient instance.

        Args:
            client_id (int): An ID, which uniquely identifies the client.
            model (torch.nn.Module): The local model that is to be trained.
            local_control_variate (torch.nn.Module): The control variate of the client to account for the client drift.
            optimizer_kind (Literal['sgd', 'adam']): The kind of optimizer that is to be used in the local training process.
            local_batchsize (int): The size of mini-batches that are to be used during training.
            local_epochs (int): The number of local epochs the client is trained.
            learning_rate (float): The learning rate that is used during local training.
            weight_decay (float): The rate at which the weights are decayed during optimization.
            momentum (float): The momentum of the optimizer.
            device (str | torch.device): The device on which the local model is to be trained.
            loss_function_kind (Literal['mse', 'nll']): The kind of loss function that is to be used for the training process.
            data_class_instance (Dataset): The local training data loader on which the model is to be trained.
            save_best_model (bool): If the switch is set, the best local model is saved. Defaults to False.

        Raises:
            ValueError: If the specified optimizer or loss function are not supported, an exception is raised.
        """

        self.client_id = client_id
        self.model = model
        self.local_control_variate = local_control_variate
        self.optimizer_kind = optimizer_kind
        self.local_batchsize = local_batchsize
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.weight_decay = weight_decay
        self.device = device
        self.data_class_instance = data_class_instance
        self.number_of_local_samples = len(data_class_instance.partitioned_training_data[self.client_id])
        self.save_best = save_best_model

        # Creates the loss function for training
        self.loss_function: torch.nn.MSELoss | torch.nn.NLLLoss
        if loss_function_kind == 'mse':
            self.loss_function = torch.nn.MSELoss().to(device)
        elif loss_function_kind == 'nll':
            self.loss_function = torch.nn.NLLLoss().to(device)
        else:
            raise ValueError(f'The loss function "{loss_function_kind}" is not supported.')

    def initialize_model(self, model: torch.nn.Module) -> None:
        """Initializes the model for the client used for training.

        Args:
            model(torch.nn.Module): The local model as a torch.nn.module.
        """
        self.model = model
        self.model.to(self.device)

    def get_state_dict(self) -> dict[str, Any]:
        """Retrieves the state dict of the client model.

        Returns:
            dict[str, Any]: The state dict of the local model.
        """
        return self.model.state_dict()

    def get_count(self) -> int:
        """ Retrieves the number of data points assigned to the client.

        Returns:
           int: The number of local training data points.
        """
        return self.number_of_local_samples

    def get_net(self) -> torch.nn.Module:
        """ Retrieves the the client model.

        Returns:
            torch.nn.Module: The local model as a torch.nn.module.
        """
        return self.model

    def set_state_dict(self, state_dict: dict[str, Any]) -> None:
        """ Sets the state dict of the local model.

            Args:
                state_dict (dict[str, Any]): The dictionary to be used to set the state_dict of the client class.
        """
        self.model.load_state_dict(state_dict)

    def set_learning_rate(self, learning_rate_from_central_server: float) -> None:
        """ Sets the learning rate received from the central server for the client for local training.

            Args:
                learning_rate_from_central_server (float): The learning rate received from the central server.

            Returns: Updated learning rate for local training fo the client.
        """
        self.learning_rate = learning_rate_from_central_server

    def get_client_id(self) -> int:
        """ Retrieves the client id.

        Returns:
           int: The unique client id.
        """
        return self.client_id

    def train(
        self,
        global_control_variate: torch.nn.Module,
        global_round: int,
    ) -> tuple[float, list[float], dict[str, Any]]:
        """ Trains the client model for specified amount of local epochs.

        Args:
            global_control_variate (torch.nn.Module): The global control variate to account for the client drift.
            global_round (int): The global round the training process is currently in.

        Returns:
            tuple[float, list[float], dict[str, Any]]: The average loss over all local training epochs and a list encoding the average loss for each
                epoch. Further the difference between the old and updated local control variate is returned.
        Raises:
            ValueError: If the specified optimizer or loss function are not supported, an exception is raised.

        """

        # Pushes the model to the device and set it in training mode
        self.model.to(self.device)
        self.model.train()

        # Creates the optimizer for training
        optimizer: torch.optim.SGD | torch.optim.Adam
        if self.optimizer_kind == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer_kind == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'The optimizer "{ self.optimizer_kind}" is not supported.')

        epoch_loss: list[float] = []
        iterations = 0

        global_control_variate_parameter = global_control_variate.state_dict()
        local_control_variate_parameter = copy.deepcopy(self.local_control_variate.state_dict())
        current_global_model = copy.deepcopy(self.model)

        # Performs local training
        for current_epoch in range(self.local_epochs):
            batch_loss = []
            # Creates the data loader for local training
            local_training_data_loader = self.data_class_instance.get_training_data_loader(
                batch_size=self.local_batchsize,
                client_id=self.client_id,
                shuffle_samples=True
            )
            # Wrap the training_data_loader with tqdm to create a progress bar
            progress_bar_training_data_loader = tqdm(
                local_training_data_loader,
                total=len(local_training_data_loader),
                desc=f"Global Round: {global_round+1} | Local Epoch: {current_epoch + 1}/{self.local_epochs} ",
            )
            for (images, labels) in progress_bar_training_data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.model.zero_grad()
                log_probability = self.model(images)
                loss = self.loss_function(log_probability, labels)
                loss.backward()
                optimizer.step()

                # Accounts for the client drift
                client_model_parameter = self.model.state_dict()
                for key in client_model_parameter:
                    client_model_parameter[key] = \
                        client_model_parameter[key] - \
                        self.learning_rate * (global_control_variate_parameter[key] - local_control_variate_parameter[key])
                self.model.load_state_dict(client_model_parameter)
                batch_loss.append(loss.item())
                iterations += 1

                # Update the progress bar postfix with the current loss value
                progress_bar_training_data_loader.set_postfix_str(f'Local Loss": "{loss.item():.6f}')

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # Updates the local control variate parameters
        new_local_control_variate_parameter = copy.deepcopy(self.local_control_variate.state_dict())
        c_delta_parameter = copy.deepcopy(self.local_control_variate.state_dict())
        global_model_parameter = current_global_model.state_dict()
        client_model_parameter = self.model.state_dict()
        # Compute c_local - c_global + (1 / (iterations * learning_rate)) * (global_model - client_model)
        for key in client_model_parameter:
            new_local_control_variate_parameter[key] = new_local_control_variate_parameter[key] - global_control_variate_parameter[key] + \
                (global_model_parameter[key] - client_model_parameter[key]) / (iterations * self.learning_rate)
            c_delta_parameter[key] = new_local_control_variate_parameter[key] - local_control_variate_parameter[key]
        self.local_control_variate.load_state_dict(new_local_control_variate_parameter)

        # Returns the average loss over all local epochs
        return sum(epoch_loss) / len(epoch_loss), epoch_loss, c_delta_parameter

    @torch.no_grad()
    def compute_train_accuracy(self) -> float:
        """ Computes the train accuracy of the client.
        Returns:
            float: The average train accuracy over all local training epochs.
        """

        # Pushes the model to the device and sets it in evaluation mode
        self.model.to(self.device)
        self.model.eval()

        number_of_correct_predictions = 0.0
        number_of_labels_total = 0.0

        # Creates the data loader for testing
        local_training_test_data_loader = self.data_class_instance.get_training_data_loader(
            batch_size=self.local_batchsize,
            client_id=self.client_id,
            shuffle_samples=False
        )

        # Wraps the training_data_loader with tqdm to create a progress bar
        progress_bar_test_data_loader = tqdm(
            local_training_test_data_loader,
            total=len(local_training_test_data_loader),
            desc="Compute Average Train Accuracy: "
        )

        # Performs inference
        for (images, labels) in progress_bar_test_data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            _, predicted_labels = torch.max(outputs, dim=1)
            predicted_labels = predicted_labels.view(-1)
            number_of_correct_predictions += torch.sum(torch.eq(predicted_labels, labels)).item()
            number_of_labels_total += len(labels)

        accuracy = 100. * number_of_correct_predictions / number_of_labels_total
        return accuracy

    @torch.no_grad()
    def compute_accuracy_and_loss_on_updated_global_model(self) -> tuple[float, float]:
        """
        Computes the accuracy and average loss of the model on the global test dataset.
        Returns:
            tuple[float, float]: A tuple containing the accuracy percentage and the average loss
            computed over all test batches. The accuracy is given as a percentage of correct
            predictions out of total predictions, and the average loss is computed as the sum
            of losses over all batches divided by the number of batches.
        """

        # Pushes the model to the device and sets it in evaluation mode
        self.model.to(self.device)
        self.model.eval()

        number_of_correct_predictions = 0.0
        number_of_labels_total = 0.0
        total_loss = 0.0

        # Creates the data loader for testing
        local_training_test_data_loader = self.data_class_instance.get_validation_data_loader(
            batch_size=self.local_batchsize
        )

        # Wraps the training_data_loader with tqdm to create a progress bar
        progress_bar_test_data_loader = tqdm(
            local_training_test_data_loader,
            total=len(local_training_test_data_loader),
            desc="Compute Average Validation Accuracy and Validation Loss: "
        )

        # Performs inference and calculates loss
        for (images, labels) in progress_bar_test_data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)

            loss = self.loss_function(outputs, labels)
            total_loss += loss.item()

            _, predicted_labels = torch.max(outputs, dim=1)
            predicted_labels = predicted_labels.view(-1)
            number_of_correct_predictions += torch.sum(torch.eq(predicted_labels, labels)).item()
            number_of_labels_total += len(labels)

        accuracy = 100. * number_of_correct_predictions / number_of_labels_total
        average_loss = total_loss / len(local_training_test_data_loader)

        return accuracy, average_loss
