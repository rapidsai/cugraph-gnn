# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Iterator, Tuple, Dict

from cugraph_pyg.utils.imports import import_optional

# Prevent PyTorch from being imported and causing an OOM error
torch = import_optional("torch")
cudf = import_optional("cudf")


class BufferedSampleReader:
    def __init__(
        self,
        nodes_call_groups: list["torch.Tensor"],
        sample_fn: Callable[..., Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]],
        *args,
        **kwargs,
    ):
        self.__sample_args = args
        self.__sample_kwargs = kwargs

        self.__nodes_call_groups = iter(nodes_call_groups)
        self.__sample_fn = sample_fn
        self.__current_call_id = 0
        self.__current_reader = None

    def __next__(self) -> Tuple[Dict[str, "torch.Tensor"], int, int]:
        new_reader = False

        if self.__current_reader is None:
            new_reader = True
        else:
            try:
                out = next(self.__current_reader)
            except StopIteration:
                new_reader = True

        if new_reader:
            # Will trigger StopIteration if there are no more call groups
            self.__current_reader = self.__sample_fn(
                self.__current_call_id,
                next(self.__nodes_call_groups),
                *self.__sample_args,
                **self.__sample_kwargs,
            )

            self.__current_call_id += 1
            out = next(self.__current_reader)

        return out

    def __iter__(self) -> Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]:
        return self
