"""Process-local Windows scheduling for a browser-facing inference service.

The terminal is only the server's launcher, not its interactive UI. Explicitly
request HighQoS so hiding that terminal does not opt inference into lower QoS.
No system power plan, registry, driver setting, or other process is changed.
Importing this module has no scheduling side effects.
"""
from __future__ import annotations

import ctypes
import os

PROCESS_POWER_THROTTLING = 4
THREAD_POWER_THROTTLING = 3
EXECUTION_SPEED = 1
ABOVE_NORMAL = 0x8000
PRIORITY_NAMES = {0x40: "idle", 0x4000: "below_normal", 0x20: "normal",
                  ABOVE_NORMAL: "above_normal", 0x80: "high", 0x100: "realtime"}


class PowerState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_uint32), ("ControlMask", ctypes.c_uint32),
                ("StateMask", ctypes.c_uint32)]


def _windows_api():
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "GetCurrentProcess": ([], ctypes.c_void_p),
        "GetCurrentThread": ([], ctypes.c_void_p),
        "GetPriorityClass": ([ctypes.c_void_p], ctypes.c_uint32),
        "SetPriorityClass": ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
        "GetProcessInformation": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
        "SetProcessInformation": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
        "GetThreadInformation": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
        "SetThreadInformation": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
    }
    for name, (args, result) in signatures.items():
        function = getattr(api, name, None)
        if function is not None:
            function.argtypes, function.restype = args, result
    return api


def _read_power(api, handle, *, thread=False):
    state = PowerState(1, 0, 0)
    get = getattr(api, "GetThreadInformation" if thread else "GetProcessInformation", None)
    info_class = THREAD_POWER_THROTTLING if thread else PROCESS_POWER_THROTTLING
    if get is None or not get(handle, info_class, ctypes.byref(state), ctypes.sizeof(state)):
        return None
    return state


def _write_power(api, handle, state, *, thread=False):
    set_info = getattr(api, "SetThreadInformation" if thread else "SetProcessInformation", None)
    info_class = THREAD_POWER_THROTTLING if thread else PROCESS_POWER_THROTTLING
    return bool(set_info and set_info(handle, info_class, ctypes.byref(state), ctypes.sizeof(state)))


class WindowsScheduling:
    def __init__(self, api):
        self.api = api
        self.process = api.GetCurrentProcess()  # Pseudo-handle: never open a different PID.
        self.original_priority = api.GetPriorityClass(self.process)
        self.original_power = _read_power(api, self.process)
        self.changed_power = self.changed_priority = False
        self.thread_high_qos = None
        self.notes = []

    def enable(self):
        previous = self.original_power or PowerState(1, 0, 0)
        # Select the speed mechanism, then explicitly disable throttling. Other
        # mechanisms (e.g. timer resolution) retain their original mask/state.
        power = PowerState(1, previous.ControlMask | EXECUTION_SPEED, previous.StateMask & ~EXECUTION_SPEED)
        self.changed_power = _write_power(self.api, self.process, power)
        if not self.changed_power:
            self.notes.append("Windows 拒绝设置进程 HighQoS，继续使用系统调度")
        if self.original_priority in (0x40, 0x4000, 0x20):
            self.changed_priority = bool(self.api.SetPriorityClass(self.process, ABOVE_NORMAL))
            if not self.changed_priority:
                self.notes.append("Windows 拒绝设置 AboveNormal，保留原优先级")
        return self.status()

    def prepare_thread(self):
        handle = self.api.GetCurrentThread()
        previous = _read_power(self.api, handle, thread=True) or PowerState(1, 0, 0)
        power = PowerState(1, previous.ControlMask | EXECUTION_SPEED, previous.StateMask & ~EXECUTION_SPEED)
        self.thread_high_qos = _write_power(self.api, handle, power, thread=True)

    def status(self):
        power = _read_power(self.api, self.process)
        disabled = bool(power is not None and power.ControlMask & EXECUTION_SPEED
                        and not power.StateMask & EXECUTION_SPEED)
        return {"mode": "responsive", "platform": "windows", "power_throttling_disabled": disabled,
                "priority": PRIORITY_NAMES.get(self.api.GetPriorityClass(self.process), "unknown"),
                "search_thread_high_qos": self.thread_high_qos, "notes": self.notes.copy()}

    def restore(self):
        if self.changed_power:
            _write_power(self.api, self.process, self.original_power or PowerState(1, 0, 0))
            self.changed_power = False
        if self.changed_priority and self.original_priority:
            self.api.SetPriorityClass(self.process, self.original_priority)
            self.changed_priority = False


_policy = None
_status = {"mode": "system", "platform": os.name, "power_throttling_disabled": False,
           "priority": "system", "search_thread_high_qos": None, "notes": []}


def configure_runtime(mode="responsive"):
    """Called by the Web CLI before any search/PyTorch threads are started."""
    global _policy, _status
    if mode not in ("responsive", "system"):
        raise ValueError("未知调度模式")
    restore_runtime()
    _status = {"mode": mode, "platform": os.name, "power_throttling_disabled": False,
               "priority": "system", "search_thread_high_qos": None, "notes": []}
    if mode == "responsive" and os.name == "nt":
        try:
            _policy = WindowsScheduling(_windows_api())
            _status = _policy.enable()
        except (OSError, AttributeError) as exc:
            if _policy is not None:
                _policy.restore()
                _policy = None
            _status["notes"] = [f"无法设置 Windows 调度：{exc}"]
    return runtime_status()


def prepare_search_thread():
    if _policy is not None:
        _policy.prepare_thread()


def runtime_status():
    return _policy.status() if _policy is not None else dict(_status, notes=_status["notes"].copy())


def restore_runtime():
    global _policy, _status
    if _policy is not None:
        _policy.restore()
        _policy = None
    _status = {"mode": "system", "platform": os.name, "power_throttling_disabled": False,
               "priority": "system", "search_thread_high_qos": None, "notes": []}
