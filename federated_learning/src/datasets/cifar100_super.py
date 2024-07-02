"""Represents a module containing datasets from the CIFAR family of datasets."""

import torchvision  # type: ignore
import numpy

from federated_learning.src.datasets.dataset import Dataset, DatasetData

class CIFAR100Coarse(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # Updates labels
        coarse_labels = numpy.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        self.targets = coarse_labels[self.targets]

        # Updates classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

class Cifar100Super(Dataset):
    """Represents the CIFAR-100 truncated dataset with superclasses."""

    dataset_id = 'cifar100-super'
    """Contains a machine-readable ID that uniquely identifies the dataset."""

    def __init__(self, path: str) -> None:
        """Initializes a new Cifar100 instance.

        Args:
            path (str): The path were the CIFAR-100 dataset is stored. If the dataset could not be found, then it is automatically downloaded to the
                specified location.
        """
        # Stores the arguments
        self.path = path

        # Exposes some information about the dataset
        self.name = 'CIFAR-100-super'

        # Creates the transformation pipeline for the data augmentation and pre-processing
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(hue=0.05, saturation=0.05),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomGrayscale(p=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        ])

        # Loads the training and validation splits of the dataset and extracts their labels
        training_data = CIFAR100Coarse(root=self.path, train=True, download=True, transform=transform)
        self.labels: list[int] = training_data.targets
        self._training_data: DatasetData = training_data
        self._validation_data: DatasetData = CIFAR100Coarse(root=self.path, train=False, download=True, transform=transform)

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
        torchvision.datasets.CIFAR100(root=path, download=True)
