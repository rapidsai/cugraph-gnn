# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

from cugraph_pyg.data.graph_store import GraphStore
from cugraph_pyg.data.feature_store import (
    TensorDictFeatureStore as DEPRECATED__TensorDictFeatureStore,
    WholeFeatureStore,
)


def TensorDictFeatureStore(*args, **kwargs):
    warnings.warn(
        "TensorDictFeatureStore is deprecated.  Consider changing your "
        "workflow to launch using 'torchrun' and store data in "
        "the faster and more memory-efficient WholeFeatureStore instead.",
        FutureWarning,
    )

    return DEPRECATED__TensorDictFeatureStore(*args, **kwargs)
