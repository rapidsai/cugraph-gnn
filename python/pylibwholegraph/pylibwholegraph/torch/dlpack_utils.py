# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibwholegraph.utils.imports import import_optional

torch = import_optional("torch")


def torch_import_from_dlpack(dp):
    return torch.utils.dlpack.from_dlpack(dp.__dlpack__())
