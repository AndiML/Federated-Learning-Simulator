"""Represents a module containing datasets from the CIFAR family of datasets."""

import torchvision  # type: ignore

from federated_learning.src.datasets.dataset import Dataset, DatasetData


class Cifar10(Dataset):
    """Represents the CIFAR-10 dataset."""

    dataset_id = 'cifar10'
    """Contains a machine-readable ID that uniquely identifies the dataset."""

    def __init__(self, path: str) -> None:
        """Initializes a new Cifar10 instance.

        Args:
            path (str): The path were the CIFAR-10 dataset is stored. If the dataset could not be found, then it is automatically downloaded to the
                specified location.
        """
        # Stores the arguments
        self.path = path

        # Exposes some information about the dataset
        self.name = 'CIFAR-10'

        # Creates the transformation pipeline for the data augmentation and pre-processing
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(hue=0.05, saturation=0.05),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomGrayscale(p=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.49139968, 0.48215827, 0.44653124), std=(0.24703233, 0.24348505, 0.26158768))
        ])

        # Loads the training and validation splits of the dataset and extracts their labels
        training_data = torchvision.datasets.CIFAR10(root=self.path, train=True, download=True, transform=transform)
        self.labels: list[int] = training_data.targets
        self._training_data: DatasetData = training_data
        self._validation_data: DatasetData = torchvision.datasets.CIFAR10(root=self.path, train=False, download=True, transform=transform)

        # Note here that the call to the constructor of the parent class depends on the creation of the labels of the training data samples
        super().__init__(self.labels)

    def get_labels(self) -> list[int]:
        """Retrieves the labels of the dataset for training.

        Returns:
            list[int]: Returns a list of the labels.
        """

        return self.labels

    @property
    def training_data(self) -> DatasetData:
        """Gets the training data of the dataset.

        Returns:
            DatasetData: Returns the training data of the dataset.
        """

        return self._training_data

    @property
    def validation_data(self) -> DatasetData:
        """Gets the validation data of the dataset.

        Returns:
            DatasetData: Returns the validation data of the dataset.
        """

        return self._validation_data

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Gets the the shape of the samples.

        Returns:
            tuple[int, ...]: Returns a tuple that contains the sizes of all dimensions of the samples.
        """

        return tuple(self.training_data[0][0].shape)

    @property
    def number_of_classes(self) -> int:
        """Gets the number of distinct classes.

        Returns:
            int: Returns the number of distinct classes.
        """

        return self.distinct_classes

    @staticmethod
    def download(path: str) -> None:
        """ Downloads the Cifar10 dataset to the specified directory.

        Args:
            path(str): The path to the directory into which the Cifar10 dataset is to be downloaded.
        """
        torchvision.datasets.CIFAR10(root=path, download=True)
