# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import pytest
import os

from cugraph.utilities.utils import import_optional, MissingModule

from cugraph_pyg.data import TensorDictFeatureStore

torch = import_optional("torch")
pylibwholegraph = import_optional("pylibwholegraph")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_tensordict_feature_store_basic_api():
    feature_store = TensorDictFeatureStore()

    node_features_0 = torch.randint(128, (100, 1000))
    node_features_1 = torch.randint(256, (100, 10))

    other_features = torch.randint(1024, (10, 5))

    feature_store["node", "feat0", None] = node_features_0
    feature_store["node", "feat1", None] = node_features_1
    feature_store["other", "feat", None] = other_features

    assert (feature_store["node"]["feat0"][:] == node_features_0).all()
    assert (feature_store["node"]["feat1"][:] == node_features_1).all()
    assert (feature_store["other"]["feat"][:] == other_features).all()

    assert len(feature_store.get_all_tensor_attrs()) == 3

    del feature_store["node", "feat0", None]
    assert len(feature_store.get_all_tensor_attrs()) == 2


@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.mg
def test_wholegraph_feature_store_basic_api():
    world_size = torch.cuda.device_count()

    # simulate torchrun call
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    raise NotImplementedError("Must write this test!")

    """
    torch.multiprocessing.spawn(
        run_test_wholegraph_feature_store_rank_0,
        args=(
            world_size,
            dtype,
        ),
        nprocs=world_size,
    )"
    """
