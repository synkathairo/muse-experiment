"""Standard setuptools build; gotrain/rules.py compiles with mypyc by default.

`uv sync` (or `uv pip install .`) builds the compiled rules engine
automatically: uv installs the [build-system] requirements (setuptools,
mypy, numpy) into the isolated build env, and setup.py needs only a C
compiler on the machine (Xcode CLT on macOS, gcc on Linux).

To opt out (pure-Python install, no compiler needed):
    WEIQI_PURE_PYTHON=1 uv sync

If no C compiler is found, the build warns loudly and falls back to pure
Python -- `uv sync` never fails for lack of a compiler.

Manual rebuild after editing rules.py (`uv sync` won't notice source edits):
    python setup.py build_ext --inplace

This compiles only gotrain/rules.py (~1.6x legality scans, ~1.8x playouts
measured 2026-10-01, roughly 5-10% of total training time). The .so shadows
rules.py transparently; delete gotrain/rules.*.so to revert.

Sharp edge: mypyc's native ints reject numpy.int64 at runtime (TypeError).
The boundary call sites in gotrain/selfplay.py already wrap with int().
Neither ty nor mypy flags a new violation -- only the test suite does.
"""
import os
import shutil
import sys

from setuptools import setup


def _c_compiler():
    return (
        shutil.which("cc")
        or shutil.which("gcc")
        or shutil.which("clang")
        or shutil.which("cl")  # MSVC
    )


def _ext_modules():
    if os.environ.get("WEIQI_PURE_PYTHON"):
        return []
    try:
        from mypyc.build import mypycify
    except ImportError:
        print(
            "warning: weiqi-train: mypy not available, building pure-Python "
            "(use uv, which provides it via [build-system])",
            file=sys.stderr,
        )
        return []
    if not _c_compiler():
        print(
            "warning: weiqi-train: no C compiler found, building pure-Python "
            "(install Xcode CLT / gcc for the compiled rules engine, or set "
            "WEIQI_PURE_PYTHON=1 to silence this warning)",
            file=sys.stderr,
        )
        return []
    return mypycify(["gotrain/rules.py"], opt_level="3")


setup(
    name="weiqi-train",
    # Explicit: this dir has several top-level folders (eval/, data/, ...)
    # that setuptools' flat-layout auto-discovery mistakes for packages.
    packages=["gotrain"],
    ext_modules=_ext_modules(),
)
