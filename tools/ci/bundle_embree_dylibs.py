#!/usr/bin/env python3
"""Bundle Embree runtime libraries with the installed Python extension."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUNTIME_PATTERNS = (
    "libembree*.dylib",
    "libtbb*.dylib",
    "libtbbmalloc*.dylib",
    "libtbb12*.dylib",
)


def _collect_candidate_roots() -> list[Path]:
    roots: set[Path] = set()
    for hint in ("/usr/local/lib", "/opt/homebrew/lib"):
        roots.add(Path(hint))

    for env_name in ("embree_DIR", "EMBREE_DIR"):
        env_value = os.environ.get(env_name)
        if not env_value:
            continue
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
    subprocess.run(["install_name_tool", "-id", f"@loader_path/{dst.name}", str(dst)], check=True)


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


def _get_extension_path() -> Path:
    spec = importlib.util.find_spec("dtcc_solar.py_embree_solar")
    if spec is None or not spec.origin:
        print(
            "Unable to locate dtcc_solar.py_embree_solar extension.",
            file=sys.stderr,
        )
        sys.exit(1)
    return Path(spec.origin).resolve()


def bundle() -> None:
    module_path = _get_extension_path()
    destination = module_path.parent

    roots = _collect_candidate_roots()
    libraries: dict[str, Path] = {}
    for root in roots:
        for pattern in RUNTIME_PATTERNS:
            for lib_path in root.glob(pattern):
                libraries.setdefault(lib_path.name, lib_path)

    if not libraries:
        print("No Embree/TBB dylibs found in the expected locations.", file=sys.stderr)
        sys.exit(1)

    copied: dict[str, Path] = {}
    for name, src_path in libraries.items():
        dst_path = destination / name
        _copy_and_rewrite(src_path, dst_path)
        copied[name] = dst_path

    for path in copied.values():
        _rewrite_dependencies(path, copied)

    _rewrite_dependencies(module_path, copied)

    print("Bundled Embree/TBB dylibs:")
    for name in sorted(copied):
        print("  ", name)


if __name__ == "__main__":
    bundle()
