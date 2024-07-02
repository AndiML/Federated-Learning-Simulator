"""Represents a module that contains the descriptor for the federated-averaging command."""

from argparse import ArgumentParser

from federated_learning.commands.base import BaseCommandDescriptor
from federated_learning.src.datasets import DATASET_IDS, DEFAULT_DATASET_ID
from federated_learning.src.nn import DEFAULT_MODEL_ID, MODEL_IDS


class ScaffoldCommandDescriptor(BaseCommandDescriptor):
    """Represents the description of federated averaging algorithm command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'scaffold'

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.
        """

        return '''Applies federated averaging with accounting for client drift.'''

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
            'number_of_runs',
            type=int,
            help='The number of repetitions to perform the federated learning experiments'
        )
        # Federated Arguments
        parser.add_argument(
            '-R',
            '--number_of_training_rounds',
            type=int,
            default=10,
            help="Number of rounds of training"
        )
        parser.add_argument(
            '-n',
            '--number_of_clients',
            type=int,
            default=100,
            help="Number of users/clients available in federated setting "
        )
        parser.add_argument(
            '-f',
            '--fraction_of_clients',
            type=float,
            default=0.1,
            help='The fraction of clients/users to participate in the federated learning training'
        )
        parser.add_argument(
            '-e',
            '--local_epochs',
            type=int,
            default=3,
            help="The number of local epochs of each client"
        )
        parser.add_argument(
            '-b',
            '--local_batchsize',
            type=int,
            default=10,
            help="Local batch size of clients during local training"
            )
        parser.add_argument(
            '-B',
            '--global_batchsize',
            type=int,
            default=10,
            help="Global batch size of the central server during validation."
            )
        parser.add_argument(
            '-l',
            '--learning_rate',
            type=float,
            default=0.01,
            help='Learning rate that the clients utilize during training'
            )
        parser.add_argument(
            '-D',
            '--learning_rate_decay',
            type=float,
            default=0.95,
            help='''The learning rate is decayed exponentially during the training. This argument is the decay rate of the learning rate. A decay rate
                1.0 would result in no decay at all. Defaults to 0.95.
            '''
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
            type=float,
            default=0.0005,
            help='The rate at which the weights are decayed during optimization. Defaults to 0.96.'
        )
        # Model Arguments
        parser.add_argument(
            '-t',
            '--model_type',
            type=str,
            default=DEFAULT_MODEL_ID,
            choices=MODEL_IDS,
            help='Type of neural network architecture used for local training on the clients'
        )
        parser.add_argument(
            '-N',
            '--norm_type',
            type=str,
            default='batch_norm',
            choices=['batch_norm', 'layer_norm', 'None'],
            help='Choose regularization technique between batch- and layer-normalisation or None'
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
        parser.add_argument(
            '-i',
            '--iidness',
            type=str,
            default='vary-datasize',
            choices=['vary-datasize', 'label-imbalance', 'homogeneous'] + [f'noniid-label{x}' for x in range(1, 100)],
            help='Depending which choice is set, different label distributions can be simulated'
        )
        parser.add_argument(
            '-I',
            '--set_dirichlet_distribution_parameter',
            type=float,
            default=0.5,
            help='Sets the parameter for the dirichlet distribution to simulate non-iid data distributions among the clients.'
        )
        parser.add_argument(
            '-p',
            '--set_proximal_parameter',
            type=float,
            default=0.01,
        )
        parser.add_argument(
            '-E',
            '--eval_every_x_round',
            type=int,
            default=10,
            help='''The number of times the client performance should be evaluated.'''
        )
