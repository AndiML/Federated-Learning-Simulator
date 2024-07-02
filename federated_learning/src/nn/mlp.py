"""Represents a module containing a simple feed forward neural network."""
import torch

from federated_learning.src.nn.base_model import BaseModel


class MLP(BaseModel):
    """Represents the classical MLP model architecture for image classification."""

    model_id = 'mlp'
    """Contains a machine-readable ID that uniquely identifies the model architecture."""

    def __init__(self, input_dimension: int, output_classes: int) -> None:
        """Initializes a new MLP instance.

        Args:
            input_dimension (int): The shape of the data that is fed to the model as input.
            output_classes (int): The number of classes between which the model has to differentiate.
        """

        super(MLP, self).__init__()

        self.layer_input = torch.nn.Linear(input_dimension, 64)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.layer_hidden = torch.nn.Linear(64, output_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the model.

        Args:
            input_tensor (torch.Tensor): The input features.

        Returns:
            torch.Tensor: Returns the outputs of the model.

        Raises:
            ValueError: If the specified optimizer or loss function are not supported, an exception is raised.

        """
        # Flattens the potential multi-dimensional input tensor
        input_tensor = input_tensor.view(-1, input_tensor.shape[1]*input_tensor.shape[-2]*input_tensor.shape[-1])
        input_tensor = self.layer_input(input_tensor)
        input_tensor = self.dropout(input_tensor)
        input_tensor = self.relu(input_tensor)
        input_tensor = self.layer_hidden(input_tensor)

        output_tensor = self.softmax(input_tensor)
        if not isinstance(output_tensor, torch.Tensor):
            raise ValueError
        return output_tensor
