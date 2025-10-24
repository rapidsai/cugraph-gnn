# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
    >> from cugraph_pyg.utils.imports import import_optional
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
    >> from from cugraph_pyg.utils.imports import import_optional
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
