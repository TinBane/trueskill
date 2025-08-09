from __future__ import annotations

import os
from setuptools import setup, Extension


def build_ext_modules():
    try:
        from Cython.Build import cythonize
    except Exception:
        return []
    ext = Extension(
        name="trueskill._fastmath",
        sources=[os.path.join("trueskill", "_fastmath.pyx")],
        extra_compile_args=["-O3"],
        language="c",
    )
    return cythonize([ext], language_level=3)


if __name__ == "__main__":
    # Allows `python setup.py build_ext --inplace` locally
    setup(ext_modules=build_ext_modules())


