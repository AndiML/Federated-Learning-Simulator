""" A module, that contains the functionality to save the global model."""

import os
from datetime import datetime
from typing import Any
import torch


class ModelSaver:
    """Represents a class that saves the current global model."""

    def __init__(self, model_directory: str) -> None:
        """Initializes a instance of the ModelSaver class.

        Args:
            model_directory (str): The directory the results of the current experiments are saved in.
        """

        self.model_directory = model_directory

    def save_global_model(self, current_training_round: int, global_model_state_dict: dict[str, Any], baseline: bool = False) -> None:
        """Saves the current state of the model to a file.

        Args:
            current_training_round (int): The current epoch of the training.
            global_model_state_dict (dict[str, Any]): The state dict of the global model, that is to be saved.
            baseline (bool):  If the switch is set, the baseline model is to be saved.
        """

        file_name_baseline = 'epoch' if baseline else 'training-round'
        file_name = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-{file_name_baseline}-{current_training_round}.pt'
        global_model_file_path = os.path.join(self.model_directory, file_name)

        torch.save({
            'epoch': current_training_round,
            'model': global_model_state_dict
        }, global_model_file_path)
