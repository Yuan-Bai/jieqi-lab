"""Generated build dependencies must not become part of project test discovery."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class TestPytestDiscovery(unittest.TestCase):
    def test_data_excluded_without_hiding_project_tests_or_losing_default_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(Path(__file__).resolve().parents[1] / "pyproject.toml",
                            root / "pyproject.toml")
            for directory in (root, root / "tools"):
                directory.mkdir(exist_ok=True)
                (directory / ("test_root.py" if directory == root else "test_tools.py")).write_text(
                    "def test_project():\n    pass\n", encoding="utf-8")
            for directory in (root / "data/build_deps/Cython/Debugger/Tests", root / ".cache",
                              root / "build", root / "dist", root / "node_modules", root / "venv"):
                directory.mkdir(parents=True)
                (directory / "test_dependency.py").write_text("import gdb\n", encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                                    cwd=root, env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("test_root.py::test_project", result.stdout)
            self.assertIn("tools/test_tools.py::test_project", result.stdout.replace("\\", "/"))
            self.assertIn("2 tests collected", result.stdout)
            self.assertNotIn("test_dependency.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
