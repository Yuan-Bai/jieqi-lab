"""Build the optional rule hotspot extension without changing runtime dependencies."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cython-path", type=Path, default=ROOT / "data/build_deps")
    args = parser.parse_args()
    sys.path.insert(0, str(args.cython_path.resolve()))
    try:
        from Cython.Build import cythonize
    except ImportError:
        parser.error("install Cython first: python -m pip install --target data/build_deps Cython==3.3.0")
    from setuptools import Extension, setup

    os.chdir(ROOT)
    source = ROOT / "native/jieqi_rules.pyx"
    extensions = cythonize(
        [Extension("_jieqi_rules", [str(source)])],
        build_dir=str(ROOT / "data/native_build/cython"),
        compile_time_env={"NATIVE_SOURCE_SHA256": hashlib.sha256(source.read_bytes()).hexdigest()},
        compiler_directives={"language_level": 3, "boundscheck": True, "wraparound": True},
        force=True,
    )
    setup(
        name="jieqi-native-rules",
        ext_modules=extensions,
        script_args=["build_ext", "--inplace", "--force", "--build-temp", "data/native_build/temp",
                     "--build-lib", "data/native_build/lib"],
    )
    print("Built _jieqi_rules. New Python processes use it automatically.")


if __name__ == "__main__":
    main()
