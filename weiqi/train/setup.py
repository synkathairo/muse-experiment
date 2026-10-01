"""Standard setuptools build with optional mypyc compilation.

Pure Python is the default: `uv pip install .` (or plain pytest runs from
this directory) never needs a C compiler or mypy.

To build the compiled rules engine (opt-in):
    WEIQI_MYPYC=1 python setup.py build_ext --inplace
or, without installing mypy into your venv (ephemeral, verified):
    WEIQI_MYPYC=1 uv run --with mypy --with setuptools --with numpy --no-sync \
        --python 3.12 setup.py build_ext --inplace
(the --with flags are all load-bearing: mypyc needs setuptools to build
and numpy importable to analyze rules.py; --python pins the .so tag)

This compiles only gotrain/rules.py (~1.6x legality scans, ~1.8x playouts
measured 2026-10-01, roughly 5-10% of total training time). The .so shadows
rules.py transparently; delete gotrain/rules.*.so to revert.

Sharp edge: mypyc's native ints reject numpy.int64 at runtime (TypeError).
The boundary call sites in gotrain/selfplay.py already wrap with int().
Neither ty nor mypy flags a new violation -- only the test suite does.
"""
import os

from setuptools import setup

ext_modules = []
if os.environ.get("WEIQI_MYPYC"):
    from mypyc.build import mypycify

    ext_modules = mypycify(["gotrain/rules.py"], opt_level="3")

setup(
    name="weiqi-train",
    # Explicit: this dir has several top-level folders (eval/, data/, ...)
    # that setuptools' flat-layout auto-discovery mistakes for packages.
    packages=["gotrain"],
    ext_modules=ext_modules,
)
