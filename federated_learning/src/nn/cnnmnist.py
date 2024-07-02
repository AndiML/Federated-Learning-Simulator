"""Represents a module containing a convolutional neural network for the MNIST or MNIST-style datasets."""
import torch

from federated_learning.src.nn.base_model import BaseModel


class CNNMnist(BaseModel):
    """Represents the classical CNN model architecture for image classification on MNIST data."""

    model_id = 'cnn'
    """Contains a machine-readable ID that uniquely identifies the model architecture."""

    def __init__(self, number_of_channels: int, output_classes: int) -> None:
        """Initializes a new CNNMnist instance.

        Args:
            number_of_channels (int): The number of channels the input data requires to be processed.
            output_classes (int): The number of classes between which the model has to differentiate.
        """
        super(CNNMnist, self).__init__()

        self.conv1 = torch.nn.Conv2d(number_of_channels, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, output_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the model.

        Args:
            input_tensor (torch.Tensor): The input features.

        Returns:
            torch.Tensor: Returns the output of the model.
        """

        input_tensor = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(input_tensor), 2))
        input_tensor = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2_drop(self.conv2(input_tensor)), 2))
        input_tensor = input_tensor.view(-1, input_tensor.shape[1]*input_tensor.shape[2]*input_tensor.shape[3])
        input_tensor = torch.nn.functional.relu(self.fc1(input_tensor))
        input_tensor = torch.nn.functional.dropout(input_tensor, training=self.training)
        input_tensor = self.fc2(input_tensor)
        return torch.nn.functional.log_softmax(input_tensor, dim=1)
