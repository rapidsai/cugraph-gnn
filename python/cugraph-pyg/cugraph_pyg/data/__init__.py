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

from cugraph_pyg.data.graph_store import (
    GraphStore as DEPRECATED__OldGraphStore,
    NewGraphStore,
)

from cugraph_pyg.data.feature_store import (
    TensorDictFeatureStore as DEPRECATED__TensorDictFeatureStore,
    FeatureStore,
)

from cugraph.utilities.utils import import_optional


def GraphStore(*args, **kwargs):
    is_multi_gpu = kwargs.pop("is_multi_gpu", None)

    if is_multi_gpu is not None:
        warnings.warn(
            "The is_multi_gpu argument is deprecated."
            "In release 25.08, multi-GPU mode will be enabled automatically"
            "when there is more than one GPU worker.",
            FutureWarning,
        )

        if is_multi_gpu:
            wgth = import_optional("pylibwholegraph.torch")
            try:
                comm = wgth.get_global_communicator()
                assert comm is not None
            except:
                raise RuntimeError(
                    "WholeGraph is not initialized.  Please call "
                    "pylibwholegraph.torch.initialize.init()"
                )
            return NewGraphStore(*args, **kwargs)
        else:
            warnings.warn(
                "Running without torchrun will be deprecated in release 25.08."
            )

    return DEPRECATED__OldGraphStore(*args, **kwargs)


def WholeFeatureStore(*args, **kwargs):
    warnings.warn("WholeFeatureStore has been renamed to FeatureStore", FutureWarning)
    return FeatureStore(*args, **kwargs)


def TensorDictFeatureStore(*args, **kwargs):
    warnings.warn(
        "TensorDictFeatureStore is deprecated.  Consider changing your "
        "workflow to launch using 'torchrun' and store data in "
        "the faster and more memory-efficient WholeFeatureStore instead.",
        FutureWarning,
    )

    return DEPRECATED__TensorDictFeatureStore(*args, **kwargs)
