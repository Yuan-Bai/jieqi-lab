"""Build optional node methods separately from the attack-detection extension."""
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

    # Do not load/lock the DLL that this build is about to replace on Windows.
    os.environ["JIEQI_SEARCH_BACKEND"] = "python"
    sys.path.insert(0, str(ROOT))
    import azero as az

    os.chdir(ROOT)
    source = ROOT / "native/jieqi_search.pyx"
    extensions = cythonize(
        [Extension("_jieqi_search", [str(source)])],
        build_dir=str(ROOT / "data/native_search_build/cython"),
        compile_time_env={"NATIVE_SOURCE_SHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "NATIVE_METHODS_SHA256": az._node_methods_sha256()},
        compiler_directives={"language_level": 3, "infer_types": False, "annotation_typing": False,
                             "boundscheck": True, "wraparound": True},
        force=True,
    )
    setup(name="jieqi-native-search", ext_modules=extensions,
          script_args=["build_ext", "--inplace", "--force", "--build-temp", "data/native_search_build/temp",
                       "--build-lib", "data/native_search_build/lib"])
    print("Built _jieqi_search. New Python processes select it with JIEQI_SEARCH_BACKEND.")


if __name__ == "__main__":
    main()
