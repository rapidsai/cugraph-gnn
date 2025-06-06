# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import torch


# module-wide fixtures

# If the rapids-pytest-benchmark plugin is installed, the "gpubenchmark"
# fixture will be available automatically. Check that this fixture is available
# by trying to import rapids_pytest_benchmark, and if that fails, set
# "gpubenchmark" to the standard "benchmark" fixture provided by
# pytest-benchmark.
try:
    import rapids_pytest_benchmark  # noqa: F401
except ImportError:
    import pytest_benchmark

    gpubenchmark = pytest_benchmark.plugin.benchmark


@pytest.fixture
def basic_pyg_graph_1():
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    size = (4, 4)
    return edge_index, size


@pytest.fixture
def basic_pyg_graph_2():
    edge_index = torch.tensor(
        [
            [0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9],
            [1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0],
        ]
    )
    size = (10, 10)
    return edge_index, size


@pytest.fixture
def sample_pyg_hetero_data():
    torch.manual_seed(12345)
    raw_data_dict = {
        "v0": torch.randn(6, 3),
        "v1": torch.randn(7, 2),
        "v2": torch.randn(5, 4),
        ("v2", "e0", "v1"): torch.tensor([[0, 2, 2, 4, 4], [4, 3, 6, 0, 1]]),
        ("v1", "e1", "v1"): torch.tensor(
            [[0, 2, 2, 2, 3, 5, 5], [4, 0, 4, 5, 3, 0, 1]]
        ),
        ("v0", "e2", "v0"): torch.tensor([[0, 2, 2, 3, 5, 5], [1, 1, 5, 1, 1, 2]]),
        ("v1", "e3", "v2"): torch.tensor(
            [[0, 1, 1, 2, 4, 5, 6], [1, 2, 3, 1, 2, 2, 2]]
        ),
        ("v0", "e4", "v2"): torch.tensor([[1, 1, 3, 3, 4, 4], [1, 4, 1, 4, 0, 3]]),
    }

    # create a nested dictionary to facilitate PyG's HeteroData construction
    hetero_data_dict = {}
    for key, value in raw_data_dict.items():
        if isinstance(key, tuple):
            hetero_data_dict[key] = {"edge_index": value}
        else:
            hetero_data_dict[key] = {"x": value}

    return hetero_data_dict
