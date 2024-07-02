"""Represents a module containing a convolutional neural network for the CIFAR-10 or CIFAR-10-style datasets."""
import torch

from federated_learning.src.nn.base_model import BaseModel


class CNNCifar10(BaseModel):
    """Represents the classical LeNet model architecture for image classification on Cifar data."""

    model_id = 'cnn'
    """Contains a machine-readable ID that uniquely identifies the model architecture."""

    def __init__(self, number_of_channels: int,  output_classes: int) -> None:
        """Initializes a new CNNCifar10 instance.

        Args:
            number_of_channels (int): The number of channels the input data requires to be processed.
            output_classes (int): The number of classes between which the model has to differentiate.
        """
        super(CNNCifar10, self).__init__()

        self.conv1 = torch.nn.Conv2d(number_of_channels, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fully_connected_layer_1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fully_connected_layer_2 = torch.nn.Linear(120, 84)
        self.fully_connected_layer_3 = torch.nn.Linear(84, output_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the model.

        Args:
            input_tensor (torch.Tensor): The input features.

        Returns:
            torch.Tensor: Returns the outputs of the model.
        """
        out = self.pool(torch.nn.functional.relu(self.conv1(input_tensor)))
        out = self.pool(torch.nn.functional.relu(self.conv2(out)))
        out = out.view(-1, 16 * 5 * 5)
        out = torch.nn.functional.relu(self.fully_connected_layer_1(out))
        out = torch.nn.functional.relu(self.fully_connected_layer_2(out))
        out = torch.nn.functional.relu(self.fully_connected_layer_3(out))
        return torch.nn.functional.log_softmax(out, dim=1)
