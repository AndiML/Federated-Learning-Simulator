"""Represents a module containing CIFAR or CIFAR-style datasets."""

import os
from typing import Optional, Callable
import subprocess as sp
import logging
import hashlib
import wget  # type: ignore

import torchvision  # type: ignore
import torch
from federated_learning.src.datasets.dataset import Dataset, DatasetData

# Set the logging level for PIL.PngImagePlugin to WARNING to suppress DEBUG messages
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)


class CINIC10_base(torchvision.datasets.vision.VisionDataset):  # type: ignore
    """A custom dataset class that interfaces with torchvision's datasets, specifically designed to handle
        the CINIC-10 dataset. This class extends torchvision.datasets.VisionDataset, providing a structured
        way to load and transform the CINIC-10 images based on the specified partition. """

    PARTS = {"train", "test"}
    base_folder = "cinic-10-batches-py"
    url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    filename = "CINIC-10.tar.gz"
    tgz_sha256 = '31b095acf6d75e25a9e028bae82a07a0f94ff6b00671be2802d34ac4efa81a9e'
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    def __init__(
        self,
        root: str,
        partition: str = "train",
        transform: Optional[Callable] = None,  # type: ignore
        target_transform: Optional[Callable] = None,  # type: ignore
        download: bool = False,
    ) -> None:
        """
        Initializes a new instance of the dataset class, designed to handle the CINIC-10 dataset
        by integrating with torchvision's dataset functionalities.

        Args:
            root (str): The base directory where the dataset will be stored. If the dataset is downloaded, it will be placed in a subdirectory
                'cinic10' within this directory.
            partition (str, optional): Specifies which subset of the dataset to use. Options are 'train' or 'test'. Defaults to "train".
            transform (Optional[Callable], optional): A function/transform that takes in a PIL image and returns a transformed version.
                Defaults to None.
            target_transform (Optional[Callable], optional): A function/transform that takes in the target (label) and transforms it.
                Defaults to None.
            download (bool, optional): If set to True, the dataset will be downloaded automatically, if it is not available at the specified root
                location. Defaults to False.

        Raises:
            RuntimeError: Raises an exception if the dataset is not found or is corrupted, and cannot be automatically downloaded.
        """

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.partition = partition
        self.root: str
        self.root = os.path.join(self.root, 'cinic10')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.data = torchvision.datasets.ImageFolder(os.path.join(self.root, partition))

        self.targets = [label for _, label in self.data.samples]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Retrieves a data point and its corresponding label from the dataset at the specified index.

        Args:
            index (int): The index of the datapoint to be retrieved. This index corresponds to a specific image and label pair in the dataset.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing a transformed image as a torch.Tensor and its corresponding label as an integer.
        """
        image, target = self.data[index]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        """Retrieves the number of datapoints contained in the CINIC-10 dataset.

        Returns:
            int: The number of datapoints.
        """
        return len(self.data)

    def _check_integrity(self) -> bool:
        """ Checks the integrity of the downloaded dataset by verifying that the file exists and that its checksum matches the expected value.
        This ensures the dataset is complete and uncorrupted, preventing errors during its usage.

        Returns:
            bool: Returns True if the dataset exists at the specified download path and its checksum matches the expected value, otherwise False.
            Informative messages are printed to console in case of failures.
        """
        if not os.path.exists(self.download_path):
            print(f"Folder {self.root} does not exist")
            return False

        checksum = self.compute_sha256(self.download_path)

        if not checksum == CINIC10_base.tgz_sha256:
            print(f"{self.download_path} checksum {checksum} does not match {CINIC10_base.tgz_sha256}")
            return False
        else:
            return True

    def compute_sha256(self, filename: str, block_size: int = 4096*16) -> str:
        """"Computes the SHA-256 hash of a file. This method reads the file in blocks, making it memory-efficient even for large files. The SHA-256
            hash is a widely used cryptographic hash function that produces a fixed-size 256-bit (32-byte) hash. It's commonly used for security
            applications and data integrity checks.
        Args:
            filename (str): The path to the file whose SHA-256 hash is to be computed.
            block_size (int): The size of each block of data read from the file during the hashing process. A larger block size can lead
                to faster processing of large files but uses more memory. The default size is 65536 bytes, which is a balance Between speed and memory
                usage.

        Returns:
            str: The SHA-256 hash of the file as a hexadecimal string. This string represents the binary hash value in a readable hex format,
            which is typically used for display and comparison purposes.
        """

        sha256_hash = hashlib.sha256()
        with open(filename, "rb") as f:

            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(block_size), b""):
                sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()

    def download(self) -> None:
        """Downloads the CINIC-10 dataset """

        os.makedirs(self.root, exist_ok=True)
        if self._check_integrity():
            print("Files already downloaded and verified")
        else:
            wget.download(self.url, out=self.download_path)

        if not all(os.path.exists(os.path.join(self.root, k)) for k in CINIC10_base.PARTS):
            cwd = os.path.abspath(os.curdir)
            os.chdir(self.root)
            sp.call(["tar", "xf", CINIC10_base.filename])
            os.chdir(cwd)

    @property
    def download_path(self) -> str:
        """
        Provides the full path where the CINIC-10 dataset is expected to be downloaded or accessed.

        Returns:
            str: The absolute path constructed by joining the root directory with the dataset's standard filename. This path is where the dataset
                file is stored or should be downloaded.
        """
        return os.path.join(self.root, CINIC10_base.filename)


class Cinic10(Dataset):
    """Represents the CINIC-10 dataset, which is a mixture of the Cifar-10 and the ImageNet dataset."""

    dataset_id = 'cinic10'
    """Contains a machine-readable ID that uniquely identifies the dataset."""

    def __init__(self, path: str):
        """Initializes a new Mnist instance.

        Args:
            path (str): The path where the CINIC-10 dataset is stored. If it does not exist, it is automatically downloaded to the specified location.
        """

        # Stores the arguments
        self.path = path

        # Exposes some information about the dataset
        self.name = 'CINIC-10'

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(hue=0.05, saturation=0.05),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomGrayscale(p=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.49139968, 0.48215827, 0.44653124), std=(0.24205776, 0.23828046, 0.25874835))
        ])

        # Loads the training and validation splits of the dataset and extracts their labels
        training_data = CINIC10_base(root=self.path, partition='train', download=True, transform=transform)
        self.labels: list[int] = training_data.targets

        self._training_data: DatasetData = training_data

        self._validation_data: DatasetData = CINIC10_base(root=self.path, partition='test', download=True, transform=transform)

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

        return (3, 32, 32)

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
        CINIC10_base(root=path, download=True)
