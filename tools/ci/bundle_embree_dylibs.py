#!/usr/bin/env python3
"""Bundle Embree runtime libraries with the installed Python extension.

This script copies the Embree and TBB dylibs next to the compiled
``py_embree_solar`` module and rewrites their install names so that they can be
loaded using ``@loader_path``. It is intended to be executed in the macOS CI
job after ``dtcc-solar`` has been installed into the virtual environment.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Patterns for the dylibs that Embree needs at runtime
RUNTIME_PATTERNS = (
    "libembree*.dylib",
    "libtbb*.dylib",
    "libtbbmalloc*.dylib",
    "libtbb12*.dylib",
)


def _collect_candidate_roots() -> list[Path]:
    """Return directories that may contain Embree runtime libraries."""
    roots: set[Path] = set()

    # Default installation locations used during CI provisioning
    for hint in ("/usr/local/lib", "/opt/homebrew/lib"):
        roots.add(Path(hint))

    # If embree_DIR is present (e.g. /usr/local/lib/cmake/embree-X.Y.Z) climb up
    # to the Embree root and add plausible subdirectories.
    env_hints = [os.environ.get("embree_DIR"), os.environ.get("EMBREE_DIR")]
    for env_value in filter(None, env_hints):
        cmake_dir = Path(env_value)
        roots.add(cmake_dir)
        roots.add(cmake_dir.parent)
        roots.add(cmake_dir.parent.parent)
        roots.add(cmake_dir.parent.parent / "lib")
        roots.add(cmake_dir.parent.parent / "bin")

    return [root for root in roots if root.exists()]


def _copy_and_rewrite(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    subprocess.run(
        ["install_name_tool", "-id", f"@loader_path/{dst.name}", str(dst)],
        check=True,
    )


def _rewrite_dependencies(target: Path, available: dict[str, Path]) -> None:
    output = subprocess.check_output(["otool", "-L", str(target)], text=True)
    for line in output.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        dep_path = line.split(" ")[0]
        dep_name = Path(dep_path).name
        if dep_name in available:
            subprocess.run(
                [
                    "install_name_tool",
                    "-change",
                    dep_path,
                    f"@loader_path/{dep_name}",
                    str(target),
                ],
                check=True,
            )


def bundle() -> None:
    try:
        module = importlib.import_module("dtcc_solar.py_embree_solar")
    except ImportError as exc:  # pragma: no cover - CI diagnostic
        print(f"Failed to import py_embree_solar: {exc}", file=sys.stderr)
        sys.exit(1)

    destination = Path(module.__file__).resolve().parent

    roots = _collect_candidate_roots()
    libraries: dict[str, Path] = {}
    for root in roots:
        for pattern in RUNTIME_PATTERNS:
            for lib_path in root.glob(pattern):
                libraries.setdefault(lib_path.name, lib_path)

    if not libraries:
        print(
            "No Embree/TBB dylibs found in the expected locations.",
            file=sys.stderr,
        )
        sys.exit(1)

    copied: dict[str, Path] = {}
    for name, src_path in libraries.items():
        dst_path = destination / name
        _copy_and_rewrite(src_path, dst_path)
        copied[name] = dst_path

    for path in copied.values():
        _rewrite_dependencies(path, copied)

    _rewrite_dependencies(Path(module.__file__).resolve(), copied)

    print("Bundled Embree/TBB dylibs:")
    for name in sorted(copied):
        print("  ", name)


if __name__ == "__main__":
    bundle()
