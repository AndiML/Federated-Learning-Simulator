# Federated Learning Experiment Environment using PyTorch

This project provides an environment for conducting federated learning experiments using PyTorch. It supports various datasets and data partitioning strategies to illustrate the effectiveness of the federated learning paradigm.

## Table of Contents

- [Federated Learning Experiment Environment using PyTorch](#federated-learning-experiment-environment-using-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Data](#data)
  - [Running Baseline Models](#running-baseline-models)
    - [Non-Federated Fashion](#non-federated-fashion)
    - [Federated Averaging Algorithm](#federated-averaging-algorithm)
      - [Data Partitioning Strategies](#data-partitioning-strategies)
  - [Federated Learning Algorithms](#federated-learning-algorithms)
  - [Client Statistics](#client-statistics)
  - [Example Commands](#example-commands)
  - [Supported Datasets](#supported-datasets)
  - [Hyperparameters](#hyperparameters)
  - [References](#references)

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

Update the environment if dependencies are added:

```bash
conda env export > environment.yml
conda env update --file environment.yml --prune
```

## Data

- Download datasets manually or automatically from Torchvision:

```bash
python -m federated_learning download-torchvision-datasets <dataset_path> <dataset>
```

- Supports MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, and more.
- Use custom datasets by placing them in the data directory and writing a PyTorch dataset wrapper.

## Running Baseline Models

### Non-Federated Fashion

Example command for MNIST with MLP on CPU simulating centralized training:

```bash
python -m federated_learning central_learning  <output_path> --model=mlp --dataset=mnist --number_of_epochs=10
```

### Federated Averaging Algorithm

#### Data Partitioning Strategies

- **Homogeneous (IID):** Equal, IID data distribution.
- **Label-Imbalance:** Imbalances controlled by Dirichlet distribution.
- **Fixed Subset of Labels:** Clients have specific distinct labels.
- **Varying Sample Size:** IID data with varying sample sizes.
- **Real-World:** Partitioned based on real word user (e.g., different authors have different writing styles).

Example command for CIFAR-10 with CNN, 5 local epochs, 10 clients per round, and IID data:

```bash
python -m federated_learning fedavg <output_path> --number_of_runs=1 --model=cnn --dataset=cifar --local_epochs=10 --iidness=homo
```

## Federated Learning Algorithms

Supports various algorithms such as FedProx, FedNova, and SCAFFOLD. Use the command structure:

```bash
python -m federated_learning <algorithm-name> --output_path=<output_path> --number_of_runs=<number_of_runs> --model=<model_type> --dataset=<dataset> --local_epochs=<local_epochs> --iidness=<iidness>
```

## Client Statistics

Client statistics during training, such as global accuracy and loss, are logged and saved, evaluated every specified number of communication rounds, and written to a CSV file.

## Example Commands

Run federated averaging with CIFAR-10 on CNN, 5 local epochs, 10 clients per round, and label imbalance:

```bash
python -m federated_learning fedavg --output_path=./ --number_of_runs=1 --model=cnn --dataset=cifar --local_epochs=10 --iidness=label-imbalance
```

Run SCAFFOLD with CIFAR-10 on CNN, 5 local epochs, 10 clients per round, and IID data:

```bash
python -m federated_learning scaffold --output_path=./ --number_of_runs=1 --model=cnn --dataset=cifar --local_epochs=10 --iidness=homo
```

Run FedProx with CIFAR-10 on CNN, 5 local epochs, 10 clients per round, and non-IID label imbalance:

```bash
python -m federated_learning fedprox --output_path=./ --number_of_runs=1 --model=cnn --dataset=cifar --local_epochs=10 --iidness=label-imbalance --set_proximal_parameter=0.1
```

Example command for running FedNova with CIFAR-10 on CNN, 5 local epochs, 10 clients per round, and repeating the experiment 3 times with incrementing seeds starting at 0:

```bash
python -m federated_learning -r 0 fednova --output_path=./  -number_of_runs=3 --model=cnn --dataset=cifar --local_epochs=10 --iidness=homo --seed_start=0

```

## Supported Datasets

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- CINIC-10
- FEMNIST
- FMNIST

## Hyperparameters

- `number_of_runs`: Number of repetitions for the experiments.
- `number_of_training_rounds (-R)`: Number of training rounds.
- `number_of_clients (-n)`: Number of clients in the federated setting.
- `fraction_of_clients (-f)`: Fraction of clients participating in training.
- `local_epochs (-e)`: Number of local epochs per client.
- `local_batchsize (-b)`: Local batch size during training.
- `global_batchsize (-B)`: Global batch size during validation.
- `learning_rate (-l)`: Learning rate for training.
- `learning_rate_decay (-D)`: Decay rate of the learning rate.
- `set_momentum (-m)`: Momentum for the optimizer.
- `weight_decay (-W)`: Weight decay rate during optimization.
- `model_type (-t)`: Neural network architecture for local training.
- `use_gpu (-g)`: Use CUDA for training.
- `optimizer (-o)`: Type of optimizer (`sgd`, `adam`).
- `iidness (-i)`: Type of IID/non-IID data distribution.
- `set_dirichlet_distribution_parameter (-I)`: Parameter for Dirichlet distribution to simulate label skew.
- `set_proximal_parameter (-p)`: Proximal parameter for FedProx.
- `eval_every_x_round (-E)`: Frequency of performance evaluation of the clients with respect to the global dataset.

## References

- **FedAvg**: McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. y. (2017). [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). AISTATS.
- **FedProx**: Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127). MLSys.
- **FedNova**: Wang, J., Yurochkin, M., Sun, Y., Papailiopoulos, D., & Khazaeni, Y. (2020). [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/abs/2007.07481). ICLR.
- **SCAFFOLD**: Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S. U., & Suresh, A. T. (2020). [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378). ICML.
