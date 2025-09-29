#!/usr/bin/env python3
"""Stage Embree runtime libraries next to the Python extension."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOTS = [Path("/usr/local/lib"), Path("/opt/homebrew/lib")]
RUNTIME_PATTERNS = [
    "libembree*.dylib",
    "libtbb*.dylib",
    "libtbbmalloc*.dylib",
    "libtbb12*.dylib",
]


def _debug(message: str) -> None:
    print(f"[bundle_embree] {message}")


def _candidate_package_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        import importlib.util

        spec = importlib.util.find_spec("dtcc_solar")
        if spec and spec.submodule_search_locations:
            dirs.extend(Path(loc) for loc in spec.submodule_search_locations)
    except Exception as exc:  # pragma: no cover - diagnostic aid
        _debug(f"find_spec failed: {exc}")

    for entry in map(Path, sys.path):
        dirs.append(entry / "dtcc_solar")
    seen: list[Path] = []
    for directory in dirs:
        directory = directory.resolve()
        if directory not in seen:
            seen.append(directory)
    return seen


def _find_extension() -> Path:
    for pkg_dir in _candidate_package_dirs():
        matches = list(pkg_dir.glob("py_embree_solar*.so"))
        if matches:
            path = matches[0].resolve()
            _debug(f"Extension located at {path}")
            return path
    _debug("Search paths: " + json.dumps([str(p) for p in _candidate_package_dirs()]))
    raise SystemExit("Unable to locate py_embree_solar extension")


def _collect_runtime_roots() -> list[Path]:
    roots = set(DEFAULT_ROOTS)
    for env in ("embree_DIR", "EMBREE_DIR"):
        value = os.environ.get(env)
        if value:
            cmake = Path(value)
            roots.update(
                {
                    cmake,
                    cmake.parent,
                    cmake.parent.parent,
                    cmake.parent.parent / "lib",
                    cmake.parent.parent / "bin",
                }
            )
    existing = [root.resolve() for root in roots if root.exists()]
    _debug("Runtime roots: " + json.dumps([str(p) for p in existing]))
    return existing


def _copy_with_install_id(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    subprocess.run(["install_name_tool", "-id", f"@loader_path/{dst.name}", str(dst)], check=True)


def _rewrite_dependencies(target: Path, table: dict[str, Path]) -> None:
    output = subprocess.check_output(["otool", "-L", str(target)], text=True)
    for line in output.splitlines()[1:]:
        dep_path = line.strip().split(" ")[0]
        dep_name = Path(dep_path).name
        if dep_name in table:
            subprocess.run(
                ["install_name_tool", "-change", dep_path, f"@loader_path/{dep_name}", str(target)],
                check=True,
            )


def main() -> None:
    extension = _find_extension()
    destination = extension.parent

    libs: dict[str, Path] = {}
    for root in _collect_runtime_roots():
        for pattern in RUNTIME_PATTERNS:
            for candidate in root.glob(pattern):
                libs.setdefault(candidate.name, candidate)

    if not libs:
        raise SystemExit("No Embree/TBB dylibs discovered")

    staged: dict[str, Path] = {}
    for name, src in libs.items():
        dst = destination / name
        _copy_with_install_id(src, dst)
        staged[name] = dst

    for path in staged.values():
        _rewrite_dependencies(path, staged)
    _rewrite_dependencies(extension, staged)

    _debug("Bundled libraries: " + ", ".join(sorted(staged)))


if __name__ == "__main__":
    main()
