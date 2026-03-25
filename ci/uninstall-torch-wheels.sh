#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

pip uninstall --yes 'torch' 'torch-geometric'

# 'pytest' leaves behind some pycache files in site-packages/torch that make 'import torch'
# seem to "work" even though there's not really a package there, leading to errors like
# "module 'torch' has no attribute 'distributed'"
#
# For the sake of testing, just fully delete 'torch' from site-packages to simulate an environment
# where it was never installed.
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SITE_PACKAGES}/torch"
