#!/usr/bin/env python3
"""Bundle Embree runtime libraries without importing the extension."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOTS = [Path("/usr/local/lib"), Path("/opt/homebrew/lib")]
PATTERNS = [
    "libembree*.dylib",
    "libtbb*.dylib",
    "libtbbmalloc*.dylib",
    "libtbb12*.dylib",
]


def _debug(msg: str) -> None:
    print(f"[bundle_embree] {msg}")


def _extension_path() -> Path:
    # Check RECORD file from installed package
    candidates = []
    for root in map(Path, sys.path):
        record = root / "dtcc_solar-0.0.0.dist-info" / "RECORD"
        if record.exists():
            candidates.append(record)
    if not candidates:
        # Fallback to site-packages dtcc_solar folder search
        for root in map(Path, sys.path):
            candidate = root / "dtcc_solar" / "py_embree_solar*.so"
            matches = list(root.glob("dtcc_solar/py_embree_solar*.so"))
            if matches:
                return matches[0]
        _debug("Could not locate py_embree_solar via sys.path")
        sys.exit(1)
    record = candidates[0]
    with record.open() as fh:
        for line in fh:
            path = line.split(",", 1)[0]
            if path.startswith("dtcc_solar/py_embree_solar") and path.endswith(".so"):
                ext_path = record.parent.parent / path
                return ext_path
    _debug("RECORD file did not contain py_embree_solar entry")
    sys.exit(1)


def _collect_roots() -> list[Path]:
    roots = set(ROOTS)
    for env in ("embree_DIR", "EMBREE_DIR"):
        value = os.environ.get(env)
        if value:
            cmake = Path(value)
            roots.update({cmake, cmake.parent, cmake.parent.parent, cmake.parent.parent / "lib"})
    existing = [root for root in roots if root.exists()]
    _debug(f"Candidate roots: {json.dumps([str(r) for r in existing])}")
    return existing


def _copy_with_id(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    subprocess.run(["install_name_tool", "-id", f"@loader_path/{dst.name}", str(dst)], check=True)


def _rewrite(target: Path, table: dict[str, Path]) -> None:
    output = subprocess.check_output(["otool", "-L", str(target)], text=True)
    for line in output.splitlines()[1:]:
        dep = line.strip().split(" ")[0]
        name = Path(dep).name
        if name in table:
            subprocess.run(
                ["install_name_tool", "-change", dep, f"@loader_path/{name}", str(target)],
                check=True,
            )


def main() -> None:
    ext = _extension_path()
    dest = ext.parent
    libs: dict[str, Path] = {}
    for root in _collect_roots():
        for pattern in PATTERNS:
            for src in root.glob(pattern):
                libs.setdefault(src.name, src)
    if not libs:
        _debug("No Embree/TBB dylibs discovered")
        sys.exit(1)
    copied: dict[str, Path] = {}
    for name, src in libs.items():
        dst = dest / name
        _copy_with_id(src, dst)
        copied[name] = dst
    for path in copied.values():
        _rewrite(path, copied)
    _rewrite(ext, copied)
    _debug("Bundled libraries: " + ", ".join(sorted(copied)))


if __name__ == "__main__":
    main()
