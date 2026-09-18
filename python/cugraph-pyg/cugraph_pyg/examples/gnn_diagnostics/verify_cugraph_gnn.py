#!/usr/bin/env python3
"""
Lightweight sanity check for a cuGraph-GNN environment.

Runs import checks for key packages (torch, torch_geometric, cugraph_pyg, pylibwholegraph), reports
CUDA availability, and exits nonzero if any core imports fail. Use this to validate fresh
installations before running heavier scripts.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from typing import Any, List, Tuple

CheckResult = Tuple[str, bool, str]


def record(checks: List[CheckResult], name: str, ok: bool, detail: str = "") -> None:
    """Append a single check result with contextual detail."""
    checks.append((name, ok, detail))


def try_import(checks: List[CheckResult], module: str, note: str | None = None) -> Any:
    """
    Attempt to import a module and record the result.

    Returns the imported module on success; otherwise logs the exception string.
    """
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "unknown")
        detail = f"{module} version {version}"
        if note:
            detail = f"{detail} ({note})"
        record(checks, module, True, detail)
        return mod
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        record(checks, module, False, detail)
        return None


def main() -> int:
    """Run the import/device checks; return 0 on success, 1 on failure."""
    checks: List[CheckResult] = []

    torch = try_import(checks, "torch")
    if torch:
        # Surface CUDA status early, since most downstream libs depend on it being enabled.
        cuda_ok = torch.cuda.is_available()
        cuda_detail = "CUDA available" if cuda_ok else "CUDA not available"
        try:
            device_count = torch.cuda.device_count()
            device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            cuda_detail = f"{cuda_detail}; {device_count} device(s): {device_names}"
        except Exception as exc:  # noqa: BLE001
            cuda_detail = f"{cuda_detail}; {type(exc).__name__}: {exc}"
        record(checks, "torch.cuda", cuda_ok, cuda_detail)

    try_import(checks, "torch_geometric")

    if torch:
        # PyG depends on torch, so only check when torch import succeeded.
        try_import(checks, "torch_geometric.nn", note="core PyG layers")

    cugraph_pyg = try_import(checks, "cugraph_pyg")
    if cugraph_pyg:
        # Ensure the core data/loader APIs are importable.
        try:
            from cugraph_pyg.data import GraphStore  # type: ignore

            _ = GraphStore
            record(checks, "cugraph_pyg.data", True, "GraphStore available")
        except Exception as exc:  # noqa: BLE001
            record(checks, "cugraph_pyg.data", False, f"{type(exc).__name__}: {exc}")

        try:
            from cugraph_pyg.loader import LinkLoader, LinkNeighborLoader  # type: ignore

            _ = (LinkLoader, LinkNeighborLoader)
            detail = "LinkLoader and LinkNeighborLoader available"
            record(checks, "cugraph_pyg.loader", True, detail)
        except Exception as exc:  # noqa: BLE001
            record(checks, "cugraph_pyg.loader", False, f"{type(exc).__name__}: {exc}")

        try:
            from cugraph_pyg import sampler  # type: ignore

            _ = sampler.BaseSampler
            record(checks, "cugraph_pyg.sampler", True, "Sampler base available")
        except Exception as exc:  # noqa: BLE001
            record(checks, "cugraph_pyg.sampler", False, f"{type(exc).__name__}: {exc}")

    try_import(checks, "pylibwholegraph")

    failures = [c for c in checks if not c[1]]

    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")

    if failures:
        print(f"\n{len(failures)} check(s) failed; environment is missing required components.")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
