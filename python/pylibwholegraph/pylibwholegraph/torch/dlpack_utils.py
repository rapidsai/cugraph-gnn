# SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.utils.dlpack


def torch_import_from_dlpack(dp):
    return torch.utils.dlpack.from_dlpack(dp.__dlpack__())
