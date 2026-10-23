"""Early CLI/worker interrupt setup. No torch import or third-party dependencies."""
from __future__ import annotations

import ctypes
import os
import signal


_controller = None


class _WindowsTrainingJob:
    """Own this CLI and its descendants, including partially started spawn processes."""
    def __init__(self):
        from ctypes import wintypes as w

        class BasicLimits(ctypes.Structure):
            _fields_ = [("ProcessTime", ctypes.c_longlong), ("JobTime", ctypes.c_longlong),
                        ("LimitFlags", w.DWORD), ("MinWorkingSet", ctypes.c_size_t),
                        ("MaxWorkingSet", ctypes.c_size_t), ("ActiveProcesses", w.DWORD),
                        ("Affinity", ctypes.c_size_t), ("Priority", w.DWORD), ("Scheduling", w.DWORD)]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                        ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("Basic", BasicLimits), ("IO", IOCounters),
                        ("ProcessMemory", ctypes.c_size_t), ("JobMemory", ctypes.c_size_t),
                        ("PeakProcessMemory", ctypes.c_size_t), ("PeakJobMemory", ctypes.c_size_t)]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            "GetCurrentProcess": ([], w.HANDLE),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
            "ExitProcess": ([w.UINT], None),
            "SetConsoleCtrlHandler": ([ctypes.c_void_p, w.BOOL], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result
        self.job = self.api.CreateJobObjectW(None, None)  # Unnamed, non-inheritable handle.
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        self.registered = False
        self.callback = ctypes.WINFUNCTYPE(w.BOOL, w.DWORD)(self._console_event)
        try:
            limits = ExtendedLimits()
            limits.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.api.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.refresh_handler()
            if not self.api.AssignProcessToJobObject(self.job, self.api.GetCurrentProcess()):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            if self.registered:
                self.api.SetConsoleCtrlHandler(self.callback, False)
            self.api.CloseHandle(self.job)
            raise
        # Keep the handle/callback alive until process exit. Closing the last handle
        # terminates descendants even when the CLI is externally terminated.

    def refresh_handler(self):
        if self.registered:
            self.api.SetConsoleCtrlHandler(self.callback, False)
            self.registered = False
        self.api.SetConsoleCtrlHandler(None, False)
        if not self.api.SetConsoleCtrlHandler(self.callback, True):
            error = ctypes.get_last_error()
            if error == 6:  # No console: job cleanup still protects external stop.
                return
            raise ctypes.WinError(error)
        self.registered = True

    def _console_event(self, event):
        if event not in (0, 1):  # CTRL_C_EVENT / CTRL_BREAK_EVENT
            return False
        self.terminate(130)
        return True

    def terminate(self, exit_code):
        # No logging/stderr, multiprocessing locks, joins or executor shutdown here.
        # Windows dispatches this callback on another thread while main may be blocked.
        try:
            if not self.api.TerminateJobObject(self.job, exit_code):
                self.api.CloseHandle(self.job)
        finally:
            self.api.ExitProcess(exit_code)


def bootstrap(module_name: str) -> None:
    """Call before torch imports, also during spawn's runpy import of the entry point."""
    global _controller
    if module_name == "__mp_main__":
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
        if os.name == "nt":
            ctypes.WinDLL("kernel32", use_last_error=True).SetConsoleCtrlHandler(None, True)
    elif module_name == "__main__" and os.name == "nt" and _controller is None:
        try:
            _controller = _WindowsTrainingJob()
        except OSError as exc:
            raise RuntimeError("无法建立本次训练的 Windows 进程清理组；训练尚未开始。") from exc


def refresh_console_handler() -> None:
    if _controller is not None:
        _controller.refresh_handler()


def terminate_owned_job(exit_code: int) -> None:
    if _controller is not None:
        _controller.terminate(exit_code)
