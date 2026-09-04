# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import multiprocessing as mp
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
    _parquet_type_staging_itemsize,
    _parquet_type_to_numpy_dtype,
    create_wholememory_tensor,
    create_wholememory_tensor_from_filelist,
    destroy_wholememory_tensor,
)
from pylibwholegraph.utils.multiprocess import multiprocess_run

pyarrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")
torch = pytest.importorskip("torch")

_GPU_COUNT = None
_MAX_OVERHEAD_GROWTH = 32 * 1024 * 1024
_SCALING_DATASET_SIZES_MIB = (32, 128)


def _gpu_count():
    global _GPU_COUNT
    if _GPU_COUNT is None:
        _GPU_COUNT = max(0, wmb.fork_get_gpu_count())
    return _GPU_COUNT


def _current_rss_bytes():
    # VmRSS measures the pages and decode buffers that are actually resident
    # in host memory.
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
    memory_location,
    row_count,
    column_count,
    last_dim_size=None,
    expected_shape=None,
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
            last_dim_size,
            file_format="parquet",
            expected_entry_count=row_count,
            expected_shape=expected_shape,
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
        )
        if local_tensor.dim() == 2:
            expected = expected.reshape(-1, column_count)
        torch.testing.assert_close(local_tensor, expected)
    finally:
        if wm_tensor is not None:
            destroy_wholememory_tensor(wm_tensor)
        finalize()


def _structured_memory_worker(
    world_rank,
    world_size,
    filename,
    memory_location,
    row_count,
    column_count,
    result_queue,
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
                file_format="parquet",
                expected_entry_count=row_count,
            )
        finally:
            stop_event.set()
            monitor.join()
        peak_increase = peak_rss[0] - baseline_rss
        assert tuple(wm_tensor.shape) == (row_count, column_count)
        result_queue.put(peak_increase)
    finally:
        if wm_tensor is not None:
            destroy_wholememory_tensor(wm_tensor)
        finalize()


def _structured_reader_memory_worker(
    world_rank,
    world_size,
    filename,
    row_count,
    column_count,
    result_queue,
):
    baseline_rss = _current_rss_bytes()
    peak_rss = [baseline_rss]
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_track_peak_rss, args=(stop_event, peak_rss), daemon=True
    )
    monitor.start()
    rows_read = 0
    checksum = 0.0
    try:
        for batch in _iter_parquet_tensors(filename, torch.float32, 2, column_count):
            rows_read += batch.shape[0]
            # Consume every value so this measures an actual read rather than
            # only iterator and metadata overhead.
            checksum += batch.sum().item()
    finally:
        stop_event.set()
        monitor.join()

    peak_increase = peak_rss[0] - baseline_rss
    assert rows_read == row_count
    assert np.isfinite(checksum)
    result_queue.put(peak_increase)


def _run_memory_worker(worker, **kwargs):
    # multiprocess_run intentionally does not return child results. A queue
    # lets the parent compare measurements from fresh processes, avoiding
    # allocator state retained by one dataset from influencing the next.
    spawn_context = mp.get_context("spawn")
    result_queue = spawn_context.Queue()
    try:
        multiprocess_run(
            1,
            partial(worker, result_queue=result_queue, **kwargs),
        )
        return result_queue.get(timeout=10)
    finally:
        result_queue.close()
        result_queue.join_thread()


def _assert_bounded_scaling(measurements, path_description):
    smallest_size, smallest_overhead = measurements[0]
    # Runtime initialization, allocator arenas, and Arrow's fixed-size pools
    # vary across CI environments. Measure that fixed cost with the smallest
    # dataset, then reject only additional overhead as the input grows. This
    # tests the property needed for multi-terabyte reads: host memory overhead
    # must remain bounded independently of total input size.
    overhead_limit = smallest_overhead + _MAX_OVERHEAD_GROWTH
    for dataset_size_mib, overhead in measurements[1:]:
        assert overhead <= overhead_limit, (
            f"Parquet {path_description} host overhead grew by "
            f"{(overhead - smallest_overhead) / 2**20:.1f} MiB when the "
            f"dataset grew from {smallest_size} MiB to {dataset_size_mib} MiB; "
            f"limit is {_MAX_OVERHEAD_GROWTH / 2**20:.1f} MiB above the "
            f"{smallest_overhead / 2**20:.1f} MiB baseline"
        )


def test_bounded_scaling_uses_smallest_dataset_as_fixed_cost_baseline():
    baseline = 160 * 1024 * 1024
    _assert_bounded_scaling(
        [(32, baseline), (128, baseline + _MAX_OVERHEAD_GROWTH)],
        "test reader",
    )

    with pytest.raises(AssertionError, match="above the 160.0 MiB baseline"):
        _assert_bounded_scaling(
            [(32, baseline), (128, baseline + _MAX_OVERHEAD_GROWTH + 1)],
            "test reader",
        )


@pytest.mark.parametrize(
    ("parquet_type", "expected_dtype"),
    [
        (pyarrow.bool_(), np.bool_),
        (pyarrow.int8(), np.int8),
        (pyarrow.int16(), np.int16),
        (pyarrow.int32(), np.int32),
        (pyarrow.int64(), np.int64),
        (pyarrow.uint8(), np.uint8),
        (pyarrow.uint16(), np.uint16),
        (pyarrow.uint32(), np.uint32),
        (pyarrow.uint64(), np.uint64),
        (pyarrow.float16(), np.float16),
        (pyarrow.float32(), np.float32),
        (pyarrow.float64(), np.float64),
    ],
)
def test_parquet_type_to_numpy_dtype(parquet_type, expected_dtype):
    assert _parquet_type_to_numpy_dtype(parquet_type) == np.dtype(expected_dtype)


def test_parquet_type_to_numpy_dtype_rejects_non_numeric_type():
    with pytest.raises(ValueError, match="unsupported Parquet column type"):
        _parquet_type_to_numpy_dtype(pyarrow.string())


@pytest.mark.parametrize(
    ("parquet_type", "expected_itemsize"),
    [
        (pyarrow.bool_(), 1),
        (pyarrow.int8(), 1),
        (pyarrow.int32(), 4),
        (pyarrow.float16(), 2),
        (pyarrow.float32(), 4),
        (pyarrow.float64(), 8),
    ],
)
def test_parquet_type_staging_itemsize(parquet_type, expected_itemsize):
    assert _parquet_type_staging_itemsize(parquet_type) == expected_itemsize


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


@pytest.mark.parametrize("memory_location", ["cpu", "cuda"])
def test_create_wholememory_tensor_from_parquet_file(tmp_path, memory_location):
    gpu_count = _gpu_count()
    if gpu_count == 0:
        pytest.skip("WholeGraph structured I/O requires at least one GPU")

    row_count = 31
    column_count = 4
    expected = torch.arange(row_count * column_count, dtype=torch.float32).reshape(
        row_count, column_count
    )
    filename = tmp_path / "tensor.parquet"
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
            memory_location=memory_location,
            row_count=row_count,
            column_count=column_count,
        ),
    )


@pytest.mark.parametrize("memory_location", ["cpu", "cuda"])
@pytest.mark.parametrize("expected_shape", [None, (31, 1)])
def test_create_wholememory_tensor_from_one_column_parquet(
    tmp_path, memory_location, expected_shape
):
    if _gpu_count() == 0:
        pytest.skip("WholeGraph structured I/O requires at least one GPU")

    row_count = 31
    filename = tmp_path / "one_column.parquet"
    parquet.write_table(
        pyarrow.table({"feature": np.arange(row_count, dtype=np.float32)}),
        filename,
        row_group_size=7,
    )

    multiprocess_run(
        1,
        partial(
            _structured_io_worker,
            filename=os.fspath(filename),
            memory_location=memory_location,
            row_count=row_count,
            column_count=1,
            expected_shape=expected_shape,
        ),
    )


@pytest.mark.parametrize("memory_location", ["cpu", "cuda"])
def test_parquet_read_has_bounded_peak_host_memory(tmp_path, memory_location):
    if _gpu_count() == 0:
        pytest.skip("WholeGraph structured I/O requires at least one GPU")

    column_count = 16
    row_size = column_count * torch.tensor([], dtype=torch.float32).element_size()
    measurements = []

    # A fourfold increase in input size should increase CPU RSS only by the
    # required WholeMemory destination. The reader/conversion overhead should
    # remain approximately flat because structured input is consumed in
    # bounded batches.
    for dataset_size_mib in _SCALING_DATASET_SIZES_MIB:
        dataset_bytes = dataset_size_mib * 1024 * 1024
        row_count = dataset_bytes // row_size
        filename = tmp_path / f"tensor_{dataset_size_mib}_mib.parquet"
        _write_parquet(filename, row_count, column_count, row_group_size=64 * 1024)
        peak_increase = _run_memory_worker(
            _structured_memory_worker,
            filename=os.fspath(filename),
            memory_location=memory_location,
            row_count=row_count,
            column_count=column_count,
        )
        destination_bytes = dataset_bytes if memory_location == "cpu" else 0
        measurements.append(
            (dataset_size_mib, max(0, peak_increase - destination_bytes))
        )

    _assert_bounded_scaling(
        measurements,
        f"read into {memory_location} WholeMemory",
    )


@pytest.mark.skip(
    reason="Peak RSS includes nondeterministic process startup allocations in CI"
)
def test_parquet_reader_has_bounded_peak_host_memory(tmp_path):
    column_count = 16
    row_size = column_count * torch.tensor([], dtype=torch.float32).element_size()
    measurements = []

    for dataset_size_mib in _SCALING_DATASET_SIZES_MIB:
        dataset_bytes = dataset_size_mib * 1024 * 1024
        row_count = dataset_bytes // row_size
        filename = tmp_path / f"reader_{dataset_size_mib}_mib.parquet"
        _write_parquet(filename, row_count, column_count, row_group_size=64 * 1024)
        peak_increase = _run_memory_worker(
            _structured_reader_memory_worker,
            filename=os.fspath(filename),
            row_count=row_count,
            column_count=column_count,
        )
        measurements.append((dataset_size_mib, peak_increase))

    _assert_bounded_scaling(
        measurements,
        "reader",
    )
