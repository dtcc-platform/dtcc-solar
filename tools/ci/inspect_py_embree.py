#!/usr/bin/env python3
"""Inspect the installed py_embree_solar extension without importing it."""

from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path


def find_extension() -> Path | None:
    try:
        import importlib.util

        spec = importlib.util.find_spec("dtcc_solar")
        if spec and spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                matches = list(Path(location).resolve().glob("py_embree_solar*.so"))
                if matches:
                    return matches[0]
    except Exception:  # pragma: no cover - diagnostic aid
        pass

    for entry in map(Path, sys.path):
        pkg_dir = (entry / "dtcc_solar").resolve()
        matches = list(pkg_dir.glob("py_embree_solar*.so"))
        if matches:
            return matches[0]
    return None


def main() -> None:
    ext = find_extension()
    if not ext:
        print("py_embree_solar extension not found")
        sys.exit(0)

    print(f"py_embree_solar: {ext}")
    contents = sorted(p.name for p in ext.parent.iterdir())
    print("Contents:", json.dumps(contents, indent=2))
    subprocess.run(["otool", "-L", str(ext)], check=True)


if __name__ == "__main__":
    main()
