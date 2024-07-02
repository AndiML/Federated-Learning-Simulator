"""Represents a module containing a convolutional neural network for the FMNIST or FMNIST-style datasets."""
import torch

from federated_learning.src.nn.base_model import BaseModel


class CNNFmnist(BaseModel):
    """Represents the classical CNN model architecture for image classification on Fashion MNIST data."""

    model_id = 'cnn'
    """Contains a machine-readable ID that uniquely identifies the model architecture."""

    def __init__(self, number_of_channels: int, output_classes: int) -> None:
        """Initializes a new CNNFashion_Mnist instance.

        Args:
            number_of_channels (int): The number of channels the input data requires to be processed.
            output_classes (int): The number of classes between which the model has to differentiate.
        """
        super(CNNFmnist, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(number_of_channels, 16, kernel_size=5, padding=2),
            # torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
            # torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.fully_connected_layer = torch.nn.Linear(7*7*32, output_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the model.

        Args:
            input_tensor (torch.Tensor): The input features.

        Returns:
            torch.Tensor: Returns the outputs of the model.

        Raises:
            ValueError: If the specified optimizer or loss function are not supported, an exception is raised.
        """

        out = self.layer1(input_tensor)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fully_connected_layer(out)
        out = torch.nn.functional.log_softmax(out, dim=1)

        if not isinstance(out, torch.Tensor):
            raise ValueError

        return out
