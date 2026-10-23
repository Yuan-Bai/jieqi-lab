"""Ctrl+C during import/spawn and blocking cleanup must not leave a training process tree."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import azero as az
import training_runtime as runtime
from tools.probe_interrupt import run_probe


class TestTrainingInterrupt(unittest.TestCase):
    def test_runtime_import_is_lightweight_and_has_no_process_side_effect(self):
        result = subprocess.run([sys.executable, "-c", "import sys; import training_runtime as r; assert 'torch' not in sys.modules; assert r._controller is None"],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_regular_import_does_not_change_signals_or_attach_a_job(self):
        with mock.patch.object(runtime.signal, "signal") as signals, \
                mock.patch.object(runtime, "_WindowsTrainingJob") as jobs:
            runtime.bootstrap("azero")
        signals.assert_not_called()
        jobs.assert_not_called()

    def test_worker_masks_interrupts_before_heavy_imports(self):
        with mock.patch.object(runtime.os, "name", "posix"), \
                mock.patch.object(runtime.signal, "signal") as signals:
            runtime.bootstrap("__mp_main__")
        signals.assert_any_call(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signals.assert_any_call(signal.SIGBREAK, signal.SIG_IGN)

    def test_windows_job_setup_is_idempotent_and_failure_stops_startup(self):
        with mock.patch.object(runtime.os, "name", "nt"), mock.patch.object(runtime, "_controller", None), \
                mock.patch.object(runtime, "_WindowsTrainingJob") as jobs:
            runtime.bootstrap("__main__")
            runtime.bootstrap("__main__")
            jobs.assert_called_once()
        with mock.patch.object(runtime.os, "name", "nt"), mock.patch.object(runtime, "_controller", None), \
                mock.patch.object(runtime, "_WindowsTrainingJob", side_effect=OSError("injected")):
            with self.assertRaisesRegex(RuntimeError, "训练尚未开始"):
                runtime.bootstrap("__main__")

    def test_cleanup_continues_after_one_worker_error_and_includes_pending_spawn(self):
        broken, good, pending = mock.Mock(), mock.Mock(), mock.Mock()
        broken.terminate.side_effect = OSError("already dying")
        with mock.patch.object(az.mp, "active_children", return_value=[broken, good]), \
                mock.patch.object(az.mp.context, "get_spawning_popen", return_value=pending):
            self.assertEqual(az.stop_child_processes(), 2)
        good.terminate.assert_called_once()
        pending.terminate.assert_called_once()

    def test_fallback_always_hard_exits_even_when_cleanup_or_stderr_raises(self):
        for failure in ("cleanup", "stderr"):
            with self.subTest(failure=failure), \
                    mock.patch.object(az.tr, "terminate_owned_job"), \
                    mock.patch.object(az, "stop_child_processes", side_effect=OSError("injected") if failure == "cleanup" else None,
                                      return_value=0), \
                    mock.patch.object(az.sys, "stderr") as stderr, \
                    mock.patch.object(az.os, "_exit") as exit_process:
                if failure == "stderr":
                    stderr.write.side_effect = OSError("closed stream")
                az._sigint_handler(signal.SIGINT, None)
                exit_process.assert_called_once_with(130)

    def job_stub(self):
        job = runtime._WindowsTrainingJob.__new__(runtime._WindowsTrainingJob)
        job.api = mock.Mock()
        job.job = 17
        job.api.TerminateJobObject.return_value = True
        return job

    def test_native_callback_handles_only_ctrl_c_and_break_without_python_cleanup(self):
        for event in (0, 1):
            job = self.job_stub()
            with self.subTest(event=event):
                self.assertTrue(job._console_event(event))
                job.api.TerminateJobObject.assert_called_once_with(17, 130)
                job.api.ExitProcess.assert_called_once_with(130)
        job = self.job_stub()
        self.assertFalse(job._console_event(2))
        job.api.TerminateJobObject.assert_not_called()

    def test_native_termination_failure_closes_job_before_exit(self):
        job = self.job_stub()
        job.api.TerminateJobObject.return_value = False
        job.terminate(130)
        job.api.CloseHandle.assert_called_once_with(17)
        job.api.ExitProcess.assert_called_once_with(130)


@unittest.skipUnless(os.name == "nt" and importlib.util.find_spec("psutil"), "Windows console integration needs optional psutil")
class TestWindowsConsoleInterrupt(unittest.TestCase):
    def check_probe(self, scenario, event=0):
        with tempfile.TemporaryDirectory() as root:
            result = run_probe(Path(root) / "probe", scenario=scenario, event=event, timeout=3)
            self.assertTrue(result["exited"], result)
            self.assertEqual(result["exit_code"], 1 if event == 2 else 130, result)
            self.assertEqual(result["surviving_children"], [], result)
            self.assertFalse(result["keyboard_interrupt_traceback"], result)
            if scenario == "startup":
                self.assertTrue(result["blocked_during_spawn"], result)

    def test_ctrl_c_during_child_torch_import_and_blocked_parent_pickle_write(self):
        self.check_probe("startup")

    def test_ctrl_break_during_running_worker(self):
        self.check_probe("active", event=1)

    def test_ctrl_c_does_not_wait_for_stderr(self):
        self.check_probe("stderr_locked")

    def test_external_parent_stop_also_terminates_owned_workers(self):
        self.check_probe("active", event=2)


if __name__ == "__main__":
    unittest.main()
