"""Process ownership for an evaluation pool; never stop unrelated children."""
from __future__ import annotations

import ctypes
import os
import signal
import time


_worker_job = None


def isolate_worker() -> None:
    """Keep external engines in this worker's process group / Windows job."""
    global _worker_job
    if os.name != "nt":
        os.setsid()
        return
    if _worker_job is not None:
        return
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

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
        "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
        "GetCurrentProcess": ([], w.HANDLE),
        "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
        "CloseHandle": ([w.HANDLE], w.BOOL),
    }
    for name, (args, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes, function.restype = args, result
    job = api.CreateJobObjectW(None, None)  # Non-inheritable: engines cannot keep the job alive.
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        limits = ExtendedLimits()
        limits.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not api.AssignProcessToJobObject(job, api.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
    except BaseException:
        api.CloseHandle(job)
        raise
    # Retain until worker exit. OS handle cleanup also kills engines after a crash
    # or forced termination, when Python finally blocks cannot run.
    _worker_job = (api, job)


def abort_pool(executor, error: BaseException, timeout: float = 3.0) -> None:
    """Cancel and kill only this pool, with bounded cleanup and original error intact."""
    # Python 3.11 has no public terminate_workers API. Snapshot the owned handles
    # before shutdown clears these fields; do not use multiprocessing.active_children().
    workers = tuple((executor._processes or {}).values())
    manager = executor._executor_manager_thread
    problems = []
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception as cleanup_error:
        problems.append(str(cleanup_error))
    for worker in workers:
        if os.name != "nt":
            try:
                # Each initialized worker is a session leader. Include its engines,
                # even if the pool manager has already terminated the worker itself.
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Worker may still be importing, before isolate_worker().
            except OSError as cleanup_error:
                problems.append(str(cleanup_error))
        try:
            worker.kill()  # Also covers interruption before worker initialization.
        except (OSError, ValueError) as cleanup_error:
            problems.append(str(cleanup_error))
    deadline = time.monotonic() + timeout
    for worker in workers:
        try:
            worker.join(max(0.0, deadline - time.monotonic()))
            if worker.is_alive():
                problems.append(f"evaluation worker {worker.pid} did not exit")
        except (OSError, ValueError) as cleanup_error:
            problems.append(str(cleanup_error))
    if manager is not None:
        try:
            manager.join(max(0.0, deadline - time.monotonic()))
            if manager.is_alive():
                problems.append("evaluation pool manager did not exit")
        except RuntimeError as cleanup_error:
            problems.append(str(cleanup_error))
    for problem in problems:
        error.add_note("Evaluation cleanup: " + problem)
