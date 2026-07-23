# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pylibwholegraph.torch.tensor import (
    _get_filelist_entry_count,
    _iter_parquet_tensors,
    _iter_pytorch_tensors,
    _iter_structured_tensors,
    _resolve_file_format,
    create_wholememory_tensor_from_filelist,
)

torch = pytest.importorskip("torch")


def _collect_structured_tensors(*args, **kwargs):
    """Materialize tiny test inputs; production code consumes these lazily."""
    return torch.cat(list(_iter_structured_tensors(*args, **kwargs)))


def test_resolve_file_format():
    assert _resolve_file_format(["tensor.pt"], "auto") == "pytorch"
    assert _resolve_file_format(["tensor.pth"], "auto") == "pytorch"
    assert _resolve_file_format(["tensor.parquet"], "auto") == "parquet"
    assert _resolve_file_format(["tensor.bin"], "auto") == "binary"

    with pytest.raises(ValueError, match="same format"):
        _resolve_file_format(["tensor.pt", "tensor.parquet"], "auto")


@pytest.mark.parametrize("use_dictionary", [False, True])
def test_load_pytorch_tensor(tmp_path, use_dictionary):
    expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    value = {"features": expected} if use_dictionary else expected
    filename = tmp_path / "tensor.pt"
    torch.save(value, filename)

    actual = _collect_structured_tensors(str(filename), "pytorch", torch.float32, 2, 3)

    torch.testing.assert_close(actual, expected)


def test_get_pytorch_filelist_entry_count(tmp_path):
    filenames = []
    for index, row_count in enumerate([4, 6]):
        filename = tmp_path / f"tensor_{index}.pt"
        torch.save(torch.rand(row_count, 3), filename)
        filenames.append(str(filename))

    assert _get_filelist_entry_count(filenames, "pytorch", torch.float32, 3) == 10


def test_load_pytorch_tensor_row_slice(tmp_path):
    expected = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    filename = tmp_path / "tensor.pt"
    torch.save(expected, filename)

    actual = _collect_structured_tensors(
        str(filename), "pytorch", torch.float32, 2, 3, 3, 7
    )

    torch.testing.assert_close(actual, expected[3:7])


def test_load_parquet_tensor(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    expected = np.arange(12, dtype=np.float32).reshape(4, 3)
    filename = tmp_path / "tensor.parquet"
    parquet.write_table(
        pyarrow.table({f"feature_{i}": expected[:, i] for i in range(3)}),
        filename,
    )

    actual = _collect_structured_tensors(str(filename), "parquet", torch.float32, 2, 3)

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


def test_iter_pytorch_tensors_reads_bounded_batches(tmp_path, monkeypatch):
    expected = torch.arange(120, dtype=torch.float64).reshape(40, 3)
    filename = tmp_path / "tensor.pt"
    torch.save(expected, filename)
    monkeypatch.setattr("pylibwholegraph.torch.tensor._STRUCTURED_BATCH_SIZE_BYTES", 48)

    batches = list(
        _iter_pytorch_tensors(
            str(filename), torch.float32, 2, 3, row_start=2, row_end=38
        )
    )

    assert len(batches) > 1
    assert max(batch.numel() * batch.element_size() for batch in batches) <= 48
    torch.testing.assert_close(torch.cat(batches), expected[2:38].to(torch.float32))


def test_legacy_pytorch_file_requires_conversion(tmp_path):
    filename = tmp_path / "legacy.pt"
    torch.save(
        torch.arange(12),
        filename,
        _use_new_zipfile_serialization=False,
    )

    with pytest.raises(ValueError, match="cannot be memory-mapped"):
        _get_filelist_entry_count([str(filename)], "pytorch", torch.int64, 0)


def test_parquet_rejects_non_numeric_columns(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    filename = tmp_path / "tensor.parquet"
    parquet.write_table(pyarrow.table({"name": ["a", "b"]}), filename)

    with pytest.raises(ValueError, match="scalar numeric columns"):
        _get_filelist_entry_count([str(filename)], "parquet", torch.float32, 1)


def test_structured_tensor_shape_validation(tmp_path):
    filename = tmp_path / "tensor.pt"
    torch.save(torch.rand(4, 2), filename)

    with pytest.raises(ValueError, match=r"expected shape \(N, 3\)"):
        list(_iter_structured_tensors(str(filename), "pytorch", torch.float32, 2, 3))
