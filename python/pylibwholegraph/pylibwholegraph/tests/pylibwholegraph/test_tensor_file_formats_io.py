# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import resource
import sys
import threading
from functools import partial

import numpy as np
import pytest

import pylibwholegraph.binding.wholememory_binding as wmb
from pylibwholegraph.torch.initialize import (
    finalize,
    init_torch_env_and_create_wm_comm,
)
from pylibwholegraph.torch.tensor import (
    _iter_structured_tensors,
    create_wholememory_tensor,
    create_wholememory_tensor_from_filelist,
    destroy_wholememory_tensor,
)
from pylibwholegraph.utils.multiprocess import multiprocess_run

pyarrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")
torch = pytest.importorskip("torch")

_GPU_COUNT = None


def _gpu_count():
    global _GPU_COUNT
    if _GPU_COUNT is None:
        _GPU_COUNT = max(0, wmb.fork_get_gpu_count())
    return _GPU_COUNT


def _current_rss_bytes():
    # mmap intentionally reserves virtual address space for an entire PyTorch
    # file, so VMS is not a useful memory-regression signal. VmRSS measures the
    # pages and decode buffers that are actually resident in host memory.
    if sys.platform.startswith("linux"):
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak_rss if sys.platform == "darwin" else peak_rss * 1024


def _track_peak_rss(stop_event, peak_rss):
    while not stop_event.wait(0.01):
        peak_rss[0] = max(peak_rss[0], _current_rss_bytes())
    peak_rss[0] = max(peak_rss[0], _current_rss_bytes())


def _structured_io_worker(
    world_rank,
    world_size,
    filename,
    file_format,
    memory_location,
    row_count,
    column_count,
):
    comm, _ = init_torch_env_and_create_wm_comm(
        world_rank, world_size, world_rank, world_size
    )
    wm_tensor = None
    try:
        wm_tensor = create_wholememory_tensor_from_filelist(
            comm,
            "distributed",
            memory_location,
            [filename],
            torch.float32,
            column_count,
            file_format=file_format,
            expected_entry_count=row_count,
        )
        # Read host WholeMemory through its native CPU view so this validates
        # the host-to-host path rather than a CUDA mapping of host allocation.
        local_tensor, local_start = wm_tensor.get_local_tensor(
            host_view=memory_location == "cpu"
        )
        local_end = local_start + local_tensor.shape[0]
        expected = torch.arange(
            local_start * column_count,
            local_end * column_count,
            device=local_tensor.device,
            dtype=torch.float32,
        ).reshape(-1, column_count)
        torch.testing.assert_close(local_tensor, expected)
    finally:
        if wm_tensor is not None:
            destroy_wholememory_tensor(wm_tensor)
        finalize()


def _structured_memory_worker(
    world_rank,
    world_size,
    filename,
    file_format,
    memory_location,
    row_count,
    column_count,
    max_peak_increase,
):
    comm, _ = init_torch_env_and_create_wm_comm(
        world_rank, world_size, world_rank, world_size
    )
    wm_tensor = None
    try:
        warmup_tensor = create_wholememory_tensor(
            comm,
            "distributed",
            memory_location,
            [1, column_count],
            torch.float32,
            None,
        )
        destroy_wholememory_tensor(warmup_tensor)
        if memory_location == "cuda":
            torch.cuda.synchronize()
        gc.collect()
        baseline_rss = _current_rss_bytes()
        peak_rss = [baseline_rss]
        stop_event = threading.Event()
        monitor = threading.Thread(
            target=_track_peak_rss, args=(stop_event, peak_rss), daemon=True
        )
        monitor.start()
        try:
            wm_tensor = create_wholememory_tensor_from_filelist(
                comm,
                "distributed",
                memory_location,
                [filename],
                torch.float32,
                column_count,
                file_format=file_format,
                expected_entry_count=row_count,
            )
        finally:
            stop_event.set()
            monitor.join()
        peak_increase = peak_rss[0] - baseline_rss
        assert peak_increase <= max_peak_increase, (
            f"{file_format} read into {memory_location} WholeMemory increased "
            f"peak host RSS by "
            f"{peak_increase / 2**20:.1f} MiB; "
            f"limit is {max_peak_increase / 2**20:.1f} MiB"
        )
        assert tuple(wm_tensor.shape) == (row_count, column_count)
    finally:
        if wm_tensor is not None:
            destroy_wholememory_tensor(wm_tensor)
        finalize()


def _structured_reader_memory_worker(
    world_rank,
    world_size,
    filename,
    file_format,
    row_count,
    column_count,
    max_peak_increase,
):
    baseline_rss = _current_rss_bytes()
    peak_rss = [baseline_rss]
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_track_peak_rss, args=(stop_event, peak_rss), daemon=True
    )
    monitor.start()
    try:
        rows_read = sum(
            batch.shape[0]
            for batch in _iter_structured_tensors(
                filename, file_format, torch.float32, 2, column_count
            )
        )
    finally:
        stop_event.set()
        monitor.join()

    peak_increase = peak_rss[0] - baseline_rss
    assert rows_read == row_count
    assert peak_increase <= max_peak_increase, (
        f"{file_format} reader increased peak host RSS by "
        f"{peak_increase / 2**20:.1f} MiB; "
        f"limit is {max_peak_increase / 2**20:.1f} MiB"
    )


def _write_parquet(filename, row_count, column_count, row_group_size):
    schema = pyarrow.schema(
        [(f"feature_{index}", pyarrow.float32()) for index in range(column_count)]
    )
    rng = np.random.default_rng(42)
    with parquet.ParquetWriter(filename, schema, compression="NONE") as writer:
        for row_start in range(0, row_count, row_group_size):
            batch_rows = min(row_group_size, row_count - row_start)
            values = rng.random((batch_rows, column_count), dtype=np.float32)
            writer.write_table(
                pyarrow.table(
                    {
                        f"feature_{index}": values[:, index]
                        for index in range(column_count)
                    },
                    schema=schema,
                )
            )


def _write_pytorch(filename, row_count, column_count):
    torch.save(torch.rand((row_count, column_count), dtype=torch.float32), filename)


def _write_structured_file(filename, file_format, row_count, column_count):
    if file_format == "parquet":
        _write_parquet(filename, row_count, column_count, row_group_size=64 * 1024)
    else:
        _write_pytorch(filename, row_count, column_count)


@pytest.mark.parametrize("memory_location", ["cpu", "cuda"])
@pytest.mark.parametrize("file_format", ["pytorch", "parquet"])
def test_create_wholememory_tensor_from_structured_file(
    tmp_path, file_format, memory_location
):
    gpu_count = _gpu_count()
    if gpu_count == 0:
        pytest.skip("WholeGraph structured I/O requires at least one GPU")

    row_count = 31
    column_count = 4
    expected = torch.arange(row_count * column_count, dtype=torch.float32).reshape(
        row_count, column_count
    )
    filename = tmp_path / f"tensor.{'pt' if file_format == 'pytorch' else 'parquet'}"
    if file_format == "pytorch":
        torch.save(expected, filename)
    else:
        parquet.write_table(
            pyarrow.table(
                {
                    f"feature_{index}": expected[:, index].numpy()
                    for index in range(column_count)
                }
            ),
            filename,
            row_group_size=7,
        )

    multiprocess_run(
        min(gpu_count, 2),
        partial(
            _structured_io_worker,
            filename=os.fspath(filename),
            file_format=file_format,
            memory_location=memory_location,
            row_count=row_count,
            column_count=column_count,
        ),
    )


@pytest.mark.parametrize("memory_location", ["cpu", "cuda"])
@pytest.mark.parametrize("file_format", ["pytorch", "parquet"])
def test_structured_read_has_bounded_peak_host_memory(
    tmp_path, file_format, memory_location
):
    if _gpu_count() == 0:
        pytest.skip("WholeGraph structured I/O requires at least one GPU")

    row_count = 1024 * 1024
    column_count = 16
    suffix = "pt" if file_format == "pytorch" else "parquet"
    filename = tmp_path / f"large_tensor.{suffix}"
    _write_structured_file(filename, file_format, row_count, column_count)

    # CUDA WholeMemory does not contribute its allocation to host RSS. CPU
    # WholeMemory necessarily adds the 64 MiB destination to RSS, so include
    # that required storage plus the same 96 MiB ceiling for mmap pages,
    # PyArrow decoding, bounded staging batches, and allocator overhead.
    destination_bytes = (
        row_count * column_count * torch.tensor([], dtype=torch.float32).element_size()
        if memory_location == "cpu"
        else 0
    )
    max_peak_increase = destination_bytes + 96 * 1024 * 1024
    multiprocess_run(
        1,
        partial(
            _structured_memory_worker,
            filename=os.fspath(filename),
            file_format=file_format,
            memory_location=memory_location,
            row_count=row_count,
            column_count=column_count,
            max_peak_increase=max_peak_increase,
        ),
    )


@pytest.mark.parametrize("file_format", ["pytorch", "parquet"])
def test_structured_reader_has_bounded_peak_host_memory(tmp_path, file_format):
    row_count = 1024 * 1024
    column_count = 16
    suffix = "pt" if file_format == "pytorch" else "parquet"
    filename = tmp_path / f"large_tensor.{suffix}"
    _write_structured_file(filename, file_format, row_count, column_count)

    multiprocess_run(
        1,
        partial(
            _structured_reader_memory_worker,
            filename=os.fspath(filename),
            file_format=file_format,
            row_count=row_count,
            column_count=column_count,
            max_peak_increase=96 * 1024 * 1024,
        ),
    )
