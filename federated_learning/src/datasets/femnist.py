"""Represents a module containing MNIST or MNIST-style datasets."""

import os

import torch
import torchvision  # type: ignore

from federated_learning.src.datasets.dataset import Dataset, DatasetData
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.datasets.utils import check_integrity


from torchvision.datasets import MNIST, utils
from PIL import Image
import os.path
import torch


class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz',
         'a8a28afae0e007f1acb87e37919a21db')]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        super(MNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
             raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))
        self.data = torch.Tensor(self.data)
        self.targets = torch.Tensor(self.targets)
        self.user_ids = torch.load(f'{self.root}/FEMNIST/processed/femnist_user_keys.pt')

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if img.dim() == 3 and img.shape[0] == 1:
            img = img.squeeze(0)  # This removes the channel dimension if it's 1

        # Reshape flat array to 2D (if necessary)
        if img.dim() == 1:
            img = img.view(28, 28)
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self)-> None:
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        os.makedirs(self.raw_folder)
        os.makedirs(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        original_filenames = {
            'femnist_train.pt': self.training_file,
            'femnist_test.pt': self.test_file
        }

        # Rename and move the files to the processed directory
        for original, new_name in original_filenames.items():
            original_path = os.path.join(self.raw_folder, original)
            new_path = os.path.join(self.raw_folder, new_name)
            os.rename(original_path, new_path)  # Rename the file
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, 'femnist_user_keys.pt'),self.processed_folder)

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]+os.path.splitext(os.path.basename(url))[1]))
            for url, _ in self.resources
        )


class Femnist(Dataset):
    """Represents the classical MNIST dataset."""

    dataset_id = 'femnist'
    """Contains a machine-readable ID that uniquely identifies the dataset."""

    def __init__(self, path: str):
        """Initializes a new Mnist instance.

        Args:
            path (str): The path where the FEMNIST dataset is stored. If it does not exist, it is automatically downloaded to the specified location.
        """

        # Stores the arguments
        self.path = path

        # Exposes some information about the dataset
        self.name = 'FEMNIST'

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Loads the training and validation splits of the dataset and extracts their labels
        training_data = FEMNIST(root=self.path, train=True, download=True, transform=transform)
        self.labels: list[int] = training_data.targets

        self._user_indices_training = training_data.users_index
        self._training_data: DatasetData = training_data

        self._validation_data: DatasetData =FEMNIST(root=self.path, train=False, download=True, transform=transform)
        self._user_indices_testing = self._validation_data.users_index

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
    def user_indices_training(self) -> list[int]:
        """Retrieves the user indices

        Returns:
             list[int]: Returns the training data of the dataset.
        """

        return self._user_indices_training

    @property
    def user_indices_testing(self) -> list[int]:
        """Retrieves the user indices

        Returns:
             list[int]: Returns the training data of the dataset.
        """

        return self._user_indices_testing

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Gets the the shape of the samples.

        Returns:
            tuple[int, ...]: Returns a tuple that contains the sizes of all dimensions of the samples.
        """

        return (1, 28, 28)

    @property
    def number_of_classes(self) -> int:
        """Gets the number of distinct classes.

        Returns:
            int: Returns the number of distinct classes.
        """

        return self.distinct_classes

    @staticmethod
    def download(path: str) -> None:
        """ Downloads the mnist dataset to the specified directory.

        Args:
            path(str): The path to the directory into which the mnist dataset is to be downloaded.
        """
        FEMNIST(root=path, download=True)
