# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pylibwholegraph.torch.tensor import (
    _get_filelist_entry_count,
    _load_structured_tensor,
    _resolve_file_format,
)

torch = pytest.importorskip("torch")


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

    actual = _load_structured_tensor(str(filename), "pytorch", torch.float32, 2, 3)

    torch.testing.assert_close(actual, expected)


def test_get_pytorch_filelist_entry_count(tmp_path):
    filenames = []
    for index, row_count in enumerate([4, 6]):
        filename = tmp_path / f"tensor_{index}.pt"
        torch.save(torch.rand(row_count, 3), filename)
        filenames.append(str(filename))

    assert _get_filelist_entry_count(filenames, "pytorch", torch.float32, 3) == 10


def test_load_parquet_tensor(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    expected = np.arange(12, dtype=np.float32).reshape(4, 3)
    filename = tmp_path / "tensor.parquet"
    parquet.write_table(
        pyarrow.table({f"feature_{i}": expected[:, i] for i in range(3)}),
        filename,
    )

    actual = _load_structured_tensor(str(filename), "parquet", torch.float32, 2, 3)

    torch.testing.assert_close(actual, torch.from_numpy(expected))


def test_structured_tensor_shape_validation(tmp_path):
    filename = tmp_path / "tensor.pt"
    torch.save(torch.rand(4, 2), filename)

    with pytest.raises(ValueError, match=r"expected shape \(N, 3\)"):
        _load_structured_tensor(str(filename), "pytorch", torch.float32, 2, 3)
