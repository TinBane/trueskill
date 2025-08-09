from __future__ import annotations

import os
from setuptools import setup, Extension


def build_ext_modules():
    # Prefer Cython if available; otherwise, compile from generated C file
    pyx_path = os.path.join("trueskill", "_fastmath.pyx")
    c_path = os.path.join("trueskill", "_fastmath.c")
    try:
        from Cython.Build import cythonize  # type: ignore
        ext = Extension(
            name="trueskill._fastmath",
            sources=[pyx_path],
            extra_compile_args=["-O3"],
            language="c",
        )
        return cythonize([ext], language_level=3)
    except Exception:
        if os.path.exists(c_path):
            return [
                Extension(
                    name="trueskill._fastmath",
                    sources=[c_path],
                    extra_compile_args=["-O3"],
                    language="c",
                )
            ]
        return []


if __name__ == "__main__":
    # Allows `python setup.py build_ext --inplace` locally
    setup(ext_modules=build_ext_modules())


