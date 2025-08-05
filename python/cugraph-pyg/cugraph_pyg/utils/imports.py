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

from packaging.requirements import Requirement
from importlib import import_module


def package_available(requirement: str) -> bool:
    """Check if a package is installed and meets the version requirement."""
    req = Requirement(requirement)
    try:
        pkg = import_module(req.name)
    except ImportError:
        return False

    if len(req.specifier) > 0:
        if hasattr(pkg, "__version__"):
            return pkg.__version__ in req.specifier
        else:
            return False

    return True


class MissingModule:
    """
    Raises RuntimeError when any attribute is accessed on instances of this
    class.

    Instances of this class are returned by import_optional() when a module
    cannot be found, which allows for code to import optional dependencies, and
    have only the code paths that use the module affected.
    """

    def __init__(self, mod_name):
        self.name = mod_name

    def __getattr__(self, attr):
        raise RuntimeError(f"This feature requires the {self.name} " "package/module")


def import_optional(mod, default_mod_class=MissingModule):
    """
    import the "optional" module 'mod' and return the module object or object.
    If the import raises ModuleNotFoundError, returns an instance of
    default_mod_class.

    This method was written to support importing "optional" dependencies so
    code can be written to run even if the dependency is not installed.

    Example
    -------
    >> from cugraph.utils import import_optional
    >> nx = import_optional("networkx")  # networkx is not installed
    >> G = nx.Graph()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      ...
    RuntimeError: This feature requires the networkx package/module

    Example
    -------
    >> class CuDFFallback:
    ..   def __init__(self, mod_name):
    ..     assert mod_name == "cudf"
    ..     warnings.warn("cudf could not be imported, using pandas instead!")
    ..   def __getattr__(self, attr):
    ..     import pandas
    ..     return getattr(pandas, attr)
    ...
    >> from cugraph.utils import import_optional
    >> df_mod = import_optional("cudf", default_mod_class=CuDFFallback)
    <stdin>:4: UserWarning: cudf could not be imported, using pandas instead!
    >> df = df_mod.DataFrame()
    >> df
    Empty DataFrame
    Columns: []
    Index: []
    >> type(df)
    <class 'pandas.core.frame.DataFrame'>
    >>
    """
    try:
        return import_module(mod)
    except ModuleNotFoundError:
        return default_mod_class(mod_name=mod)
