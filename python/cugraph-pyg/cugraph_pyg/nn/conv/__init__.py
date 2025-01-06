# Copyright (c) 2023-2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

HAVE_CUGRAPH_OPS = False
try:
    import pylibcugraphops

    HAVE_CUGRAPH_OPS = True
except ImportError:
    pass
except Exception as e:
    warnings.warn(f"Unexpected error while importing pylibcugraphops: {e}")

if HAVE_CUGRAPH_OPS:
    from .gat_conv import GATConv
    from .gatv2_conv import GATv2Conv
    from .hetero_gat_conv import HeteroGATConv
    from .rgcn_conv import RGCNConv
    from .sage_conv import SAGEConv
    from .transformer_conv import TransformerConv

    __all__ = [
        "GATConv",
        "GATv2Conv",
        "HeteroGATConv",
        "RGCNConv",
        "SAGEConv",
        "TransformerConv",
    ]
