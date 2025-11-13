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

from cugraph_pyg.utils.imports import import_optional, MissingModule

from cugraph_pyg.data import FeatureStore

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")
pylibwholegraph = import_optional("pylibwholegraph")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_feature_store_basic_api(single_pytorch_worker):
    feature_store = FeatureStore()

    node_features_0 = torch.randint(128, (100, 1000))
    node_features_1 = torch.randint(256, (100, 10))

    other_features = torch.randint(1024, (10, 5))

    feature_store["node", "feat0", None] = node_features_0
    feature_store["node", "feat1", None] = node_features_1
    feature_store["other", "feat", None] = other_features

    assert (
        feature_store["node", "feat0", None].get_local_tensor().cpu() == node_features_0
    ).all()
    assert (
        feature_store["node", "feat1", None].get_local_tensor().cpu() == node_features_1
    ).all()
    assert (
        feature_store["other", "feat", None].get_local_tensor().cpu() == other_features
    ).all()

    ixr = torch.randperm(node_features_0.shape[0])
    assert (
        feature_store["node", "feat0", None][ixr].cpu() == node_features_0[ixr]
    ).all()

    assert len(feature_store.get_all_tensor_attrs()) == 3

    del feature_store["node", "feat0", None]
    assert len(feature_store.get_all_tensor_attrs()) == 2


@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float64,
    ],
)
def test_feature_store_basic_api_types(single_pytorch_worker, dtype):
    features = torch.arange(0, 2000)
    features = features.reshape((features.numel() // 100, 100)).to(dtype)

    whole_store = FeatureStore()
    whole_store["node", "fea", None] = features

    ix = torch.arange(features.shape[0])
    assert (whole_store["node", "fea", None][ix].cpu() == features[ix]).all()
    assert (whole_store["node", "fea", None][ix].cpu() == features[ix]).all()

    ix = torch.randperm(features.shape[0])

    label = torch.arange(0, features.shape[0]).reshape((features.shape[0], 1))
    whole_store["node", "label", None] = label

    assert (whole_store["node", "label", None][ix].cpu() == label[ix]).all()
