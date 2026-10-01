#!/usr/bin/env python3
"""Opt-in mypyc build for the rules engine.

Compiles gotrain/rules.py to a native extension with mypyc. Measured
2026-10-01: ~1.6x faster legality scans, ~1.8x faster playouts, roughly
5-10% of total training time. The type annotations in rules.py are the
only prerequisite and are already in place.

Usage (run from weiqi/train/):
    uv pip install mypy      # once per machine; mypyc ships with mypy.
                             # Needs a C compiler: gcc on Linux, Xcode CLT on macOS.
    python build_rules_ext.py

The .so shadows rules.py transparently: every existing import keeps
working with no code changes. Verify after building:
    python -m pytest tests/ -q

Sharp edges (read before adopting):
- mypyc's native ints are stricter than Python's: callers must pass an
  exact `int`, not `numpy.int64`. The boundary call sites in
  gotrain/selfplay.py already wrap with int(); any NEW call site that
  hands a numpy scalar to a rules method will fail loudly at runtime.
  Type checkers (ty, mypy) do NOT catch this -- only the test suite does.
- Only rules.py is compiled. The rest of the package is torch/numpy-bound
  or too dynamic for mypyc to help.
- The .so is platform-specific: each machine builds its own. Never commit it.

To revert: rm gotrain/rules.*.so gotrain/rules__mypyc.*.so
"""
from __future__ import annotations

import os
import shutil
import sys


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    if not os.path.isfile(os.path.join("gotrain", "rules.py")):
        print("error: run this script from weiqi/train/ (gotrain/rules.py not found)",
              file=sys.stderr)
        return 1
    try:
        from mypyc.build import mypycify
    except ImportError:
        print("error: mypyc not found. Install it with:  uv pip install mypy",
              file=sys.stderr)
        return 1
    from setuptools import setup

    # packages=[] disables setuptools' flat-layout auto-discovery
    # (this dir has several top-level folders); we only build the ext.
    setup(
        name="gotrain-rules-compiled",
        packages=[],
        ext_modules=mypycify(["gotrain/rules.py"], opt_level="3"),
        script_args=["build_ext", "--inplace"],
    )
    shutil.rmtree("build", ignore_errors=True)
    print("built: gotrain/rules.*.so (shadows rules.py; delete it to revert)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
