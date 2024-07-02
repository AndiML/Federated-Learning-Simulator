"""Represents a module that contains the descriptor for the central learning command."""

from argparse import ArgumentParser

from federated_learning.commands.base import BaseCommandDescriptor
from federated_learning.src.datasets import DATASET_IDS, DEFAULT_DATASET_ID
from federated_learning.src.nn import MODEL_IDS, DEFAULT_MODEL_ID


class CentralLearningCommandDescriptor(BaseCommandDescriptor):
    """Represents the description of central learning command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'central-learning'

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.
        """

        return '''Evaluates the baseline performance of the model.'''

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            parser (ArgumentParser): The command line argument parser to which the arguments are to be added.
        """

        parser.add_argument(
            'output_path',
            type=str,
            help='The path to the directory into which the results of the experiments are saved.'
        )
        parser.add_argument(
            'dataset_path',
            type=str,
            help='The path to the directory into which the specified dataset is retrieved or downloaded.'
        )
        parser.add_argument(
            '-e',
            '--number_of_epochs',
            type=int,
            default=10,
            help="The number of epochs for the model to train."
        )
        parser.add_argument(
            '-b',
            '--batchsize',
            type=int,
            default=10,
            help="Batch size during training"
        )
        parser.add_argument(
            '-l',
            '--learning_rate',
            type=float,
            default=0.01,
            help='Learning rate that is used during training'
        )
        parser.add_argument(
            '-D',
            '--learning_rate_decay',
            type=float,
            default=1.0,
            help='''The learning rate is decayed exponentially during the training. This argument is the decay rate of the learning rate. A decay rate
                1.0 would result in no decay at all. Defaults to 1.0.'''
        )
        parser.add_argument(
            '-m',
            '--set_momentum',
            type=float,
            default=0.9,
            help='Sets the level of momentum for specified optimizer'
        )
        parser.add_argument(
            '-W',
            '--weight_decay',
            dest='weight_decay',
            type=float,
            default=0.0005,
            help='The rate at which the weights are decayed during optimization. Defaults to 0.96.'
        )
        # model arguments
        parser.add_argument(
            '-t',
            '--model_type',
            type=str,
            default=DEFAULT_MODEL_ID,
            choices=MODEL_IDS,
            help='Type of neural network architecture used for local training on the clients'
        )
        parser.add_argument(
            '-L',
            '--loss_function',
            type=str,
            default='nll',
            choices=['mse', 'nll'],
            help="Type of loss function"
        )
        parser.add_argument(
            '-N',
            '--norm_type',
            type=str,
            default='batch_norm',
            choices=['batch_norm', 'layer_norm', 'None'],
            help='Choose regularization technique between batch and layer normalization or None'
        )
        parser.add_argument(
            '-d',
            '--dataset',
            type=str,
            default=DEFAULT_DATASET_ID,
            choices=DATASET_IDS,
            help=f'Name of dataset used in the federated training process. Defaults to "{DEFAULT_DATASET_ID}".'
        )
        parser.add_argument(
            '-g',
            '--use_gpu',
            action='store_true',
            help="If the switch is set, cuda is utilized for the federated learning process."
        )
        parser.add_argument(
            '-o',
            '--optimizer',
            type=str,
            default='sgd',
            choices=['sgd', 'adam'],
            help="Type of optimizer"
        )
