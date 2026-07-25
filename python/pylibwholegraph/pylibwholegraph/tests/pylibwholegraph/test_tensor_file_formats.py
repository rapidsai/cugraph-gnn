# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pylibwholegraph.torch.tensor import (
    _get_filelist_entry_count,
    _iter_parquet_tensors,
    _resolve_file_format,
    create_wholememory_tensor_from_filelist,
)

torch = pytest.importorskip("torch")


def _collect_parquet_tensors(*args, **kwargs):
    """Materialize tiny test inputs; production code consumes these lazily."""
    return torch.cat(list(_iter_parquet_tensors(*args, **kwargs)))


def test_resolve_file_format():
    assert _resolve_file_format(["tensor.parquet"], "auto") == "parquet"
    assert _resolve_file_format(["tensor.bin"], "auto") == "binary"

    with pytest.raises(ValueError, match="same format"):
        _resolve_file_format(["tensor.bin", "tensor.parquet"], "auto")

    with pytest.raises(ValueError, match="PyTorch files are not supported"):
        _resolve_file_format(["tensor.pt"], "auto")

    with pytest.raises(ValueError, match="must be one of"):
        _resolve_file_format(["tensor.pt"], "pytorch")


def test_load_parquet_tensor(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    expected = np.arange(12, dtype=np.float32).reshape(4, 3)
    filename = tmp_path / "tensor.parquet"
    parquet.write_table(
        pyarrow.table({f"feature_{i}": expected[:, i] for i in range(3)}),
        filename,
    )

    actual = _collect_parquet_tensors(str(filename), torch.float32, 2, 3)

    torch.testing.assert_close(actual, torch.from_numpy(expected))


def test_get_parquet_filelist_entry_count_uses_metadata(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    filenames = []
    for index, row_count in enumerate([4, 6]):
        filename = tmp_path / f"tensor_{index}.parquet"
        parquet.write_table(
            pyarrow.table(
                {
                    f"feature_{i}": np.arange(row_count, dtype=np.float32)
                    for i in range(3)
                }
            ),
            filename,
        )
        filenames.append(str(filename))

    assert _get_filelist_entry_count(filenames, "parquet", torch.float32, 3) == 10

    with pytest.raises(ValueError, match="expected_entry_count is 11"):
        create_wholememory_tensor_from_filelist(
            None,
            "distributed",
            "cuda",
            filenames,
            torch.float32,
            3,
            file_format="parquet",
            expected_entry_count=11,
        )


def test_iter_parquet_tensors_reads_requested_rows(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    expected = np.arange(36, dtype=np.float32).reshape(12, 3)
    filename = tmp_path / "tensor.parquet"
    parquet.write_table(
        pyarrow.table({f"feature_{i}": expected[:, i] for i in range(3)}),
        filename,
        row_group_size=3,
    )

    batches = list(
        _iter_parquet_tensors(
            str(filename), torch.float32, 2, 3, row_start=2, row_end=10
        )
    )

    torch.testing.assert_close(torch.cat(batches), torch.from_numpy(expected[2:10]))


def test_parquet_rejects_non_numeric_columns(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    filename = tmp_path / "tensor.parquet"
    parquet.write_table(pyarrow.table({"name": ["a", "b"]}), filename)

    with pytest.raises(ValueError, match="scalar numeric columns"):
        _get_filelist_entry_count([str(filename)], "parquet", torch.float32, 1)


def test_parquet_tensor_shape_validation(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    filename = tmp_path / "tensor.parquet"
    parquet.write_table(
        pyarrow.table(
            {
                "feature_0": np.arange(4, dtype=np.float32),
                "feature_1": np.arange(4, dtype=np.float32),
            }
        ),
        filename,
    )
    with pytest.raises(ValueError, match=r"expected shape \(N, 3\)"):
        list(_iter_parquet_tensors(str(filename), torch.float32, 2, 3))
