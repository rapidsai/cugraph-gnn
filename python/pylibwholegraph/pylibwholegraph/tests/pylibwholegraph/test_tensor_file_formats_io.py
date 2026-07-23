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
    _iter_parquet_tensors,
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
            "cuda",
            [filename],
            torch.float32,
            column_count,
            file_format=file_format,
            expected_entry_count=row_count,
        )
        local_tensor, local_start = wm_tensor.get_local_tensor()
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


def _parquet_memory_worker(
    world_rank,
    world_size,
    filename,
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
            "cuda",
            [1, column_count],
            torch.float32,
            None,
        )
        destroy_wholememory_tensor(warmup_tensor)
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
                "cuda",
                [filename],
                torch.float32,
                column_count,
                file_format="parquet",
                expected_entry_count=row_count,
            )
        finally:
            stop_event.set()
            monitor.join()
        peak_increase = peak_rss[0] - baseline_rss
        assert peak_increase <= max_peak_increase, (
            f"Parquet read increased peak host RSS by {peak_increase / 2**20:.1f} MiB; "
            f"limit is {max_peak_increase / 2**20:.1f} MiB"
        )
        assert tuple(wm_tensor.shape) == (row_count, column_count)
    finally:
        if wm_tensor is not None:
            destroy_wholememory_tensor(wm_tensor)
        finalize()


def _parquet_reader_memory_worker(
    world_rank,
    world_size,
    filename,
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
            for batch in _iter_parquet_tensors(filename, torch.float32, 2, column_count)
        )
    finally:
        stop_event.set()
        monitor.join()

    peak_increase = peak_rss[0] - baseline_rss
    assert rows_read == row_count
    assert peak_increase <= max_peak_increase, (
        f"Parquet reader increased peak host RSS by {peak_increase / 2**20:.1f} MiB; "
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


@pytest.mark.parametrize("file_format", ["pytorch", "parquet"])
def test_create_wholememory_tensor_from_structured_file(tmp_path, file_format):
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
            row_count=row_count,
            column_count=column_count,
        ),
    )


def test_parquet_read_has_bounded_peak_host_memory(tmp_path):
    if _gpu_count() == 0:
        pytest.skip("WholeGraph structured I/O requires at least one GPU")

    row_count = 1024 * 1024
    column_count = 16
    filename = tmp_path / "large_tensor.parquet"
    _write_parquet(filename, row_count, column_count, row_group_size=64 * 1024)

    # The uncompressed tensor is 64 MiB. Allow 96 MiB for PyArrow, NumPy, and
    # allocator overhead. The previous full-table implementation needed at
    # least one full Arrow table plus another full column-stacked allocation.
    max_peak_increase = 96 * 1024 * 1024
    multiprocess_run(
        1,
        partial(
            _parquet_memory_worker,
            filename=os.fspath(filename),
            row_count=row_count,
            column_count=column_count,
            max_peak_increase=max_peak_increase,
        ),
    )


def test_parquet_reader_has_bounded_peak_host_memory(tmp_path):
    row_count = 1024 * 1024
    column_count = 16
    filename = tmp_path / "large_tensor.parquet"
    _write_parquet(filename, row_count, column_count, row_group_size=64 * 1024)

    multiprocess_run(
        1,
        partial(
            _parquet_reader_memory_worker,
            filename=os.fspath(filename),
            row_count=row_count,
            column_count=column_count,
            max_peak_increase=96 * 1024 * 1024,
        ),
    )
