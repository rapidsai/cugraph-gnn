# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from pylibwholegraph.utils.imports import import_optional, MissingModule

torch = import_optional("torch")

# NOTE: using more specific 'import_optional()' than just 'torch' for import-time checks
#       (e.g. those needed for defining base classes) can be helpful because 'torch' can appear
#       to be available even after a 'pip uninstall torch' if any files are left behind in
#       'site-packages/torch'.
torch_utils_data = import_optional("torch.utils.data")


if not isinstance(torch_utils_data, MissingModule):

    class NodeClassificationDataset(torch_utils_data.Dataset):
        def __init__(self, raw_dataset):
            self.dataset = raw_dataset

        def __getitem__(self, index):
            return self.dataset[index]

        def __len__(self):
            return len(self.dataset)
else:

    class NodeClassificationDataset:
        def __init__(self, raw_dataset):
            raise ModuleNotFoundError(
                "NodeClassificationDataset requires 'torch.utils.data'. Install 'torch'."
            )


def create_node_classification_datasets(data_and_label: dict):
    train_data = {
        "idx": data_and_label["train_idx"],
        "label": data_and_label["train_label"],
    }
    valid_data = {
        "idx": data_and_label["valid_idx"],
        "label": data_and_label["valid_label"],
    }
    test_data = {
        "idx": data_and_label["test_idx"],
        "label": data_and_label["test_label"],
    }
    train_dataset = list(
        list(zip(train_data["idx"], train_data["label"].astype(np.int64)))
    )
    valid_dataset = list(
        list(zip(valid_data["idx"], valid_data["label"].astype(np.int64)))
    )
    test_dataset = list(
        list(zip(test_data["idx"], test_data["label"].astype(np.int64)))
    )

    return (
        NodeClassificationDataset(train_dataset),
        NodeClassificationDataset(valid_dataset),
        NodeClassificationDataset(test_dataset),
    )


def get_train_dataloader(
    train_dataset,
    batch_size: int,
    *,
    replica_id: int = 0,
    num_replicas: int = 1,
    num_workers: int = 0,
):
    train_sampler = torch_utils_data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_replicas,
        rank=replica_id,
        shuffle=True,
        drop_last=True,
    )
    train_dataloader = torch_utils_data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else None,
        sampler=train_sampler,
    )
    return train_dataloader


def get_valid_test_dataloader(
    valid_test_dataset, batch_size: int, *, num_workers: int = 0
):
    valid_test_sampler = torch_utils_data.distributed.DistributedSampler(
        valid_test_dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False
    )
    valid_test_dataloader = torch_utils_data.DataLoader(
        valid_test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=valid_test_sampler,
    )
    return valid_test_dataloader
