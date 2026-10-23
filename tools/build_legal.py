"""Build the optional exact legal-move fast path without rebuilding attack rules."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/build_deps"))
sys.path.insert(0, str(ROOT))
os.environ["JIEQI_LEGAL_BACKEND"] = "python"

from Cython.Build import cythonize
from setuptools import Extension, setup
import jieqi as jq


def main():
    source = ROOT / "native/jieqi_legal.pyx"
    os.chdir(ROOT)
    extensions = cythonize(
        [Extension("_jieqi_legal", [str(source)])],
        build_dir=str(ROOT / "data/native_legal_build/cython"),
        compile_time_env={"NATIVE_SOURCE_SHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "NATIVE_METHODS_SHA256": jq._legal_methods_sha256()},
        compiler_directives={"language_level": 3, "infer_types": False, "annotation_typing": False,
                             "boundscheck": True, "wraparound": True},
        force=True,
    )
    setup(name="jieqi-native-legal", ext_modules=extensions,
          script_args=["build_ext", "--inplace", "--force", "--build-temp", "data/native_legal_build/temp",
                       "--build-lib", "data/native_legal_build/lib"])


if __name__ == "__main__":
    main()
