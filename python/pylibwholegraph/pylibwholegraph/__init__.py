# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibwholegraph._version import __git_commit__, __version__

# If libwholegraph was installed as a wheel, we must request it to load the
# library symbols. Otherwise, we assume that the library was installed in a
# system path that ld can find.
try:
    import libwholegraph
except ModuleNotFoundError:
    pass
else:
    libwholegraph.load_library()
    del libwholegraph
