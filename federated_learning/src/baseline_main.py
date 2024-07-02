"""A module, which contains the baseline experiments."""

import logging
from typing import Literal

import torch
from tqdm import tqdm

from federated_learning.src.datasets.dataset import Dataset
from federated_learning.src.experiments import ExperimentLogger


def train_baseline_model(
    data_class_instance: Dataset,
    model: torch.nn.Module,
    current_epoch: int,
    optimizer_kind: Literal['sgd', 'adam'],
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    loss_function_kind: Literal['mse', 'nll'],
    device: str | torch.device,
    logger: logging.Logger,
    experiment_logger: ExperimentLogger
) -> tuple[float, float, float, float]:
    """Performs the training of the baseline model.

    Args:
        data_class_instance (Dataset): The local training data loader on which the model is to be trained.
        model (torch.nn.Module): The model that is to be trained.
        current_epoch (int):  The current epoch the model is trained.
        optimizer_kind (Literal['sgd', 'adam']): The kind of optimizer that is to be used in the local training process.
        batch_size (int): The size of mini-batches that are to be used during training.
        learning_rate (float): The learning rate that is used during local training.
        momentum (float): The momentum of the optimizer.
        weight_decay (float): The rate at which the weights are decayed during optimization.
        loss_function_kind (Literal['mse', 'nll']): The kind of loss function that is to be used for the training process.
        device (str | torch.device): The device on which the local model is to be trained.
        logger (logging.Logger): Logger is provided to log metrics obtained during the federated learning process directly to the command line.
        experiment_logger (ExperimentLogger): A logger to save the experiments results and write it to the TensorBoard
    Returns:
       tuple[float, float, float, float]: The  train loss and accuracy of the model as well as validation loss and accuracy of the the baseline model
        after each epoch.
    """
    # Notifies the experiment logger that a new epoch has begun
    experiment_logger.begin_training_round(current_epoch + 1, baseline=True)

    logger.info(f'| Start Training Baseline Model Epoch {current_epoch+1}: ',  extra={'end_section': True})
    training_loss_per_epoch = train_one_epoch(
        model=model,
        data_class_instance=data_class_instance,
        optimizer_kind=optimizer_kind,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        loss_function_kind=loss_function_kind,
        device=device
    )
    training_accuracy_per_epoch = compute_train_accuracy_one_epoch(
        model=model,
        data_class_instance=data_class_instance,
        batch_size=batch_size,
        device=device
    )

    # Logs the learning rate at the end of each epoch, because the learning rate is decayed and therefore changes each epoch
    experiment_logger.add_metric('Learning Rate', 'Training Learning Rate', learning_rate)

    # Logs average training loss for one epoch
    logger.info(f"Train Loss: {training_loss_per_epoch:.5f}", extra={'start_section': True})
    experiment_logger.add_metric('Average Train Loss', 'Average Training Loss', training_loss_per_epoch)

    # Logs training accuracy for one epoch
    logger.info(f"Train Accuracy: {training_accuracy_per_epoch:.4f}", extra={'start_section': True, 'end_section': True})
    experiment_logger.add_metric('Average Train Accuracy', 'Average Training Accuracy', training_accuracy_per_epoch)

    # Validates the model after one round of training and logs the loss and accuracy
    validation_loss_per_epoch, validation_accuracy_per_epoch = validate_model(
        model=model,
        data_class_instance=data_class_instance,
        batch_size=batch_size,
        loss_function_kind=loss_function_kind,
        device=device
    )

    # Logs validation loss after one epoch
    logger.info(f"Validation Loss:  {validation_loss_per_epoch:.5f}", extra={'start_section': True})
    experiment_logger.add_metric('Average Val Loss', 'Average Validation Loss', validation_loss_per_epoch)

    # Logs validation accuracy after one epoch
    logger.info(f"Validation Accuracy: {validation_accuracy_per_epoch:.4f}", extra={'start_section': True})
    experiment_logger.add_metric('Average Val Accuracy', 'Average Validation Accuracy', validation_accuracy_per_epoch)
    # Notifies the experiment logger that the epoch has finished, which flushes out everything to disk
    experiment_logger.end_training_round()

    # Returns the loss and accuracy for training and validation for each epoch
    return training_loss_per_epoch, training_accuracy_per_epoch, validation_loss_per_epoch, validation_accuracy_per_epoch


def train_one_epoch(
    model: torch.nn.Module,
    data_class_instance: Dataset,
    optimizer_kind: Literal['sgd', 'adam'],
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    loss_function_kind: Literal['mse', 'nll'],
    device: str | torch.device,
) -> float:
    """Trains a given model for one epoch.

    Args:
        model (torch.nn.Module): The local model that is to be trained.
        data_class_instance (Dataset): The local training data loader on which the model is to be trained.
        optimizer_kind (Literal['sgd', 'adam']): The kind of optimizer that is to be used in the local training process.
        batch_size (int): The size of mini-batches that are to be used during training.
        learning_rate (float): The learning rate that is used during local training.
        momentum (float): The momentum of the optimizer.
        weight_decay (float): The rate at which the weights are decayed during optimization.
        loss_function_kind (Literal['mse', 'nll']): The kind of loss function that is to be used for the training process.
        device (str | torch.device): The device on which the local model is to be trained.

    Raises:
        ValueError: If the specified optimizer or loss function are not supported, an exception is raised.

    Returns:
        float: The average loss over one training epoch.
    """

    # Sets model in training mode
    model.train()

    # Creates the optimizer for training
    optimizer: torch.optim.SGD | torch.optim.Adam
    if optimizer_kind == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_kind == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f'The optimizer "{optimizer_kind}" is not supported.')

    # Retrieves the loss function for training
    loss_function: torch.nn.MSELoss | torch.nn.NLLLoss
    if loss_function_kind == 'mse':
        loss_function = torch.nn.MSELoss().to(device)
    elif loss_function_kind == 'nll':
        loss_function = torch.nn.NLLLoss().to(device)
    else:
        raise ValueError(f'The loss function "{loss_function_kind}" is not supported.')

    # Creates the data loader, which manages the loading of the dataset, it uses multiple worker processes, which load the dataset samples
    # asynchronously in the background (an worker initialization function and a generator are used to fix the seeds of the random number generators of
    # the worker process)
    training_data_loader = data_class_instance.get_training_data_loader(batch_size=batch_size, client_id=0, shuffle_samples=True)

    # Performs training for one epoch
    batch_loss: list[float] = []
    progress_bar_training_data_loader = tqdm(training_data_loader, total=len(training_data_loader), desc="Train for one epoch: ")
    for (images, labels) in progress_bar_training_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        model.zero_grad()
        log_probability = model(images)
        loss = loss_function(log_probability, labels)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())

        # Adds the current loss of the training as a postfix to the progress bar
        progress_bar_training_data_loader.set_postfix_str(f' Loss: {loss.item():.6f}')

    # Returns the average loss over all batches in one epoch
    return sum(batch_loss) / len(batch_loss)


@torch.no_grad()
def compute_train_accuracy_one_epoch(
    model: torch.nn.Module,
    data_class_instance: Dataset,
    batch_size: int,
    device: str | torch.device,
) -> float:
    """Computes the train accuracy for one epoch.

    Args:
        model (torch.nn.Module): The local model that is to be trained.
        data_class_instance (Dataset): The local training data loader on which the model is to be trained.
        batch_size (int): The size of mini-batches that are to be used during training.
        device (str | torch.device): The device on which the local model is to be trained.

    Returns:
        float: The average accuracy over one training epoch.
    """

    # Sets the model in evaluation mode
    model.eval()
    number_of_correct_predictions = 0.0
    number_of_labels_total = 0.0

    # Creates the data loader for testing
    test_data_loader = data_class_instance.get_training_data_loader(batch_size=batch_size, client_id=0, shuffle_samples=False)

    # Performs inference
    progress_bar_training_data_loader = tqdm(test_data_loader, total=len(test_data_loader), desc='Compute Average Train Accuracy after epoch: ')
    for (images, labels) in progress_bar_training_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted_labels = torch.max(outputs, dim=1)
        predicted_labels = predicted_labels.view(-1)
        number_of_correct_predictions += torch.sum(torch.eq(predicted_labels, labels)).item()
        number_of_labels_total += len(labels)

    accuracy = 100.0 * number_of_correct_predictions / number_of_labels_total

    # Returns the accuracy over all batches in one epoch
    return accuracy


@torch.no_grad()
def validate_model(
    model: torch.nn.Module,
    data_class_instance: Dataset,
    batch_size: int,
    loss_function_kind: Literal['mse', 'nll'],
    device: str | torch.device,
) -> tuple[float, float]:
    """Validates the model the trained model.

    Args:
        model (torch.nn.Module): The local model that is to be validated.
        data_class_instance (Dataset): The local training data loader on which the model is to be trained.
        batch_size (int): The size of mini-batches that are to be used during validation.
        loss_function_kind (Literal['mse', 'nll']): The kind of loss function that is to be used for the validation process.
        device (str | torch.device): The device on which the local model is to be validated.

    Raises:
        ValueError: If the specified loss function is not supported, an exception is raised.

    Returns:
        tuple[float, float]: The average accuracy over the entire validation dataset.
    """

    # Sets in evaluation mode
    model.eval()

    # Retrieves the loss function for validation
    loss_function: torch.nn.MSELoss | torch.nn.NLLLoss
    if loss_function_kind == 'mse':
        loss_function = torch.nn.MSELoss().to(device)
    elif loss_function_kind == 'nll':
        loss_function = torch.nn.NLLLoss().to(device)
    else:
        raise ValueError(f'The loss function "{loss_function_kind}" is not supported.')

    # Creates the validation data loader for validation
    validation_data_loader = data_class_instance.get_validation_data_loader(batch_size=batch_size)
    progress_bar_validation_data_loader = tqdm(
        validation_data_loader,
        total=len(validation_data_loader),
        desc='Compute Validation Loss and Accuracy: '
    )

    # Performs validation
    validation_loss = 0.0
    number_of_data_points_total = 0.0
    number_of_correct_predictions = 0.0
    for (images, labels) in progress_bar_validation_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        validation_loss += loss_function(outputs, labels).item()
        _, predicted_labels = torch.max(outputs, dim=1)
        predicted_labels = predicted_labels.view(-1)
        number_of_correct_predictions += torch.sum(torch.eq(predicted_labels, labels)).item()
        number_of_data_points_total += len(labels)
    validation_accuracy = 100.0 * number_of_correct_predictions / number_of_data_points_total
    # Returns the loss and accuracy of the model
    return validation_loss/number_of_data_points_total, validation_accuracy
