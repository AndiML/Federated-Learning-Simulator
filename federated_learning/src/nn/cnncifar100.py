"""Represents a module containing a convolutional neural network for the CIFAR-100 or CIFAR-100-style datasets."""
import torch

from federated_learning.src.nn.base_model import BaseModel


class CNNCifar100(BaseModel):
    """Represents the classical LeNet model architecture for image classification on Cifar data."""

    model_id = 'cnn'
    """Contains a machine-readable ID that uniquely identifies the model architecture."""

    def __init__(self, number_of_channels: int,  output_classes: int) -> None:
        """Initializes a new CNNCifar10 instance.

        Args:
            number_of_channels (int): The number of channels the input data requires to be processed.
            output_classes (int): The number of classes between which the model has to differentiate.
        """
        super(CNNCifar100, self).__init__()

        self.convolutional_layer_1 = torch.nn.Conv2d(in_channels=number_of_channels, out_channels=64, kernel_size=3, padding=1)
        self.normalization_layer_1 = torch.nn.BatchNorm2d(64)

        self.convolutional_layer_2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.normalization_layer_2 = torch.nn.BatchNorm2d(128)

        self.convolutional_layer_3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.normalization_layer_3 = torch.nn.BatchNorm2d(256)

        self.convolutional_layer_4 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.normalization_layer_4 = torch.nn.BatchNorm2d(256)

        self.convolutional_layer_5 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.normalization_layer_5 = torch.nn.BatchNorm2d(512)

        self.convolutional_layer_6 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.normalization_layer_6 = torch.nn.BatchNorm2d(512)

        self.convolutional_layer_7 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.normalization_layer_7 = torch.nn.BatchNorm2d(512)

        self.convolutional_layer_8 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.normalization_layer_8 = torch.nn.BatchNorm2d(512)

        self.fully_connected_layer_1 = torch.nn.Linear(512, 4096)
        self.fully_connected_layer_2 = torch.nn.Linear(4096, 4096)
        self.fully_connected_layer_3 = torch.nn.Linear(4096, output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the model.

        Args:
            x (torch.Tensor): The input features.

        Returns:
            torch.Tensor: Returns the outputs of the model.
        """

        y = self.convolutional_layer_1(x)
        y = self.normalization_layer_1(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)

        y = self.convolutional_layer_2(y)
        y = self.normalization_layer_2(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)

        y = self.convolutional_layer_3(y)
        y = self.normalization_layer_3(y)
        y = torch.nn.functional.relu(y)

        y = self.convolutional_layer_4(y)
        y = self.normalization_layer_4(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)

        y = self.convolutional_layer_5(y)
        y = self.normalization_layer_5(y)
        y = torch.nn.functional.relu(y)

        y = self.convolutional_layer_6(y)
        y = self.normalization_layer_6(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)

        y = self.convolutional_layer_7(y)
        y = self.normalization_layer_7(y)
        y = torch.nn.functional.relu(y)

        y = self.convolutional_layer_8(y)
        y = self.normalization_layer_8(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.max_pool2d(y, kernel_size=2, stride=2)

        y = torch.flatten(y, 1)

        y = self.fully_connected_layer_1(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.dropout(y, 0.5)

        y = self.fully_connected_layer_2(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.dropout(y, 0.5)

        y = self.fully_connected_layer_3(y)

        return torch.nn.functional.log_softmax(y, dim=1)
