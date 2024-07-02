# Federated-Learning-Experiment Environment using PyTorch

This project provides an environment for conducting federated learning experiments using PyTorch. It supports various datasets and data partitioning strategies to illustrate the effectiveness of the federated learning paradigm.

## Requirements

Before using the project, create the Conda environment:

```bash
conda env create --file environment.yml
```

Activate the environment:

```bash
conda activate federated-learning-simulator
```

Run federated learning experiments with the following command:

```bash
python -m federated_learning <arguments...>
```

Deactivate the Conda environment when finished:

```bash
conda deactivate
```

If you install any dependencies, update the environment file manually to ensure portability:

```bash
conda env export > environment.yml
```

To update the environment from a file:

```bash
conda env update --file environment.yml --prune
```

## Data

- Download train and test datasets manually or they will be automatically downloaded from Torchvision datasets using the command:

```bash
python -m federated_learning download-torchvision-datasets <dataset_path> <dataset>
```

- Experiments are run on MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, and other datasets.
- To use your own dataset: Move your dataset to the data directory and write a wrapper on the PyTorch dataset class.

## Running the Baseline Model in a Non-Federated Fashion

To run the baseline experiment with MNIST on MLP using CPU:

```bash
python -m federated_learning federated-learning-baseline ./ --model=mlp --dataset=mnist --number_of_epochs=10
```

## Running Baseline Federated Averaging Algorithm

### Data Partitioning Strategies

The data partitioning strategies implemented include:

- **Homogeneous (IID):** Data is split equally and distributed in an IID fashion.
- **Label-Imbalance:** Data distribution is controlled by a Dirichlet distribution to create imbalances.
- **Fixed Subset of Labels:** Each client has a specified number of distinct labels.
- **Varying Sample Size:** Data is distributed IID but with varying sample sizes among clients.
- **Real-World:** Real-world datasets are partitioned based on user indices.

To run federated averaging with CIFAR-10 on a CNN with 5 local epochs, 10 clients per communication round, and IID data per client:

```bash
python -m federated_learning federated-averaging ./ 1 --model=cnn --dataset=cifar --local_epochs=10 --iidness=homo
```

To run federated averaging with CIFAR-10 on a CNN with 5 local epochs, 10 clients per communication round, and only 3 labels per client:

```bash
python -m federated_learning federated-averaging ./ 1 --model=cnn --dataset=cifar --local_epochs=10 --iidness=noniid-label3
```

To run federated averaging with CIFAR-10 on a CNN with 5 local epochs, 10 clients per communication round, and varying dataset size per client:

```bash
python -m federated_learning federated-averaging ./ 1 --model=cnn --dataset=cifar --local_epochs=10 --iidness=vary-datasize
```

## Additional Federated Learning Algorithms

Apart from the standard federated averaging, other algorithms such as FedProx, FedNova, and SCAFFOLD are implemented. Each algorithm can be run with the following command structure:

```bash
python -m federated_learning <algorithm-name> ./ <number_of_runs> --model=<model_type> --dataset=<dataset> --local_epochs=<local_epochs> --iidness=<iidness>
```

For example, to run FedNova with MNIST on MLP using non-IID data:

```bash
python -m federated_learning fednova ./ 1 --model=mlp --dataset=mnist --local_epochs=10 --iidness=noniid-label3
```

## Client Statistics

Client statistics during training, such as global accuracy and loss, are logged and saved. The performance is evaluated every specified number of communication rounds, and the results are written to a CSV file.

## Example Commands

To run federated averaging with CIFAR-10 on CNN with 5 local epochs, 10 clients per communication round, and data distribution based on label imbalances:

```bash
python -m federated_learning federated-averaging ./ 1 --model=cnn --dataset=cifar --local_epochs=10 --iidness=label-imbalance
```

To run federated averaging with CIFAR-10 on CNN with 5 local epochs, 10 clients per communication round, and IID quantity skew:

```bash
python -m federated_learning federated-averaging ./ 1 --model=cnn --dataset=cifar --local_epochs=10 --iidness=homogeneous
```

## Supported Datasets

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- CINIC-10
- FEMNIST
- FMNIST
