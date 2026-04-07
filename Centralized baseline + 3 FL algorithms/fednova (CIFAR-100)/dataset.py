"""Dataloaders for the CIFAR-100 dataset."""

from typing import List, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fednova.dataset_preparation import DataPartitioner


def load_datasets(config: DictConfig) -> Tuple[List[DataLoader], DataLoader, List]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        Parameterises the dataset partitioning process
    num_clients : int
        The number of clients that hold a part of the data
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, List]
        The DataLoader for training, the DataLoader for testing, client dataset sizes.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    trainset = datasets.CIFAR100(
        root=config.datapath, train=True, download=True, transform=transform_train
    )

    testset = datasets.CIFAR100(
        root=config.datapath, train=False, download=True, transform=transform_test
    )

    partition_sizes = [1.0 / config.num_clients for _ in range(config.num_clients)]

    partition_obj = DataPartitioner(
        trainset, partition_sizes, is_non_iid=config.NIID, alpha=config.alpha
    )
    ratio = partition_obj.ratio

    trainloaders = []
    for data_split in range(config.num_clients):
        client_partition = partition_obj.use(data_split)
        trainloaders.append(
            torch.utils.data.DataLoader(
                client_partition,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=True,
            )
        )

    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # Save partition indices for MIA membership labels.
    # Keys are client ids (as strings), values are lists of CIFAR-100 train indices.
    import json, os
    partition_path = os.path.join(
        config.datapath,
        f"partition_indices_cifar100_seed{config.seed}_alpha{config.alpha}"
        f"_clients{config.num_clients}.json"
    )
    partition_indices = {
        str(i): [int(x) for x in partition_obj.partitions[i]]
        for i in range(config.num_clients)
    }
    with open(partition_path, "w") as f:
        json.dump(partition_indices, f)
    print(f"[MIA] Partition indices saved → {partition_path}")

    return trainloaders, test_loader, ratio
