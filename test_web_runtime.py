"""Win32 scheduling contracts, using a fake kernel so tests never reprioritize pytest."""
import ctypes

import pytest

import web_runtime as runtime


class Kernel:
    def __init__(self, *, priority=0x20, control=0, state=0, fail=False):
        self.priority = priority
        self.power = (control, state)
        self.thread_power = (3, 3)
        self.fail = fail
        self.writes = []
        self.priorities = []

    def GetCurrentProcess(self):
        return -1

    def GetCurrentThread(self):
        return -2

    def GetPriorityClass(self, handle):
        assert handle == -1
        return self.priority

    def SetPriorityClass(self, handle, value):
        assert handle == -1
        self.priorities.append(value)
        if self.fail:
            return False
        self.priority = value
        return True

    def GetProcessInformation(self, handle, kind, output, size):
        assert handle == -1 and kind == runtime.PROCESS_POWER_THROTTLING
        assert size == 12
        state = ctypes.cast(output, ctypes.POINTER(runtime.PowerState)).contents
        state.Version, state.ControlMask, state.StateMask = 1, *self.power
        return True

    def SetProcessInformation(self, handle, kind, source, size):
        assert handle == -1 and kind == runtime.PROCESS_POWER_THROTTLING and size == 12
        state = ctypes.cast(source, ctypes.POINTER(runtime.PowerState)).contents
        assert state.Version == 1
        self.writes.append((state.ControlMask, state.StateMask))
        if self.fail:
            return False
        self.power = state.ControlMask, state.StateMask
        return True

    def GetThreadInformation(self, handle, kind, output, size):
        assert handle == -2 and kind == runtime.THREAD_POWER_THROTTLING and size == 12
        state = ctypes.cast(output, ctypes.POINTER(runtime.PowerState)).contents
        state.Version, state.ControlMask, state.StateMask = 1, *self.thread_power
        return True

    def SetThreadInformation(self, handle, kind, source, size):
        assert handle == -2 and kind == runtime.THREAD_POWER_THROTTLING and size == 12
        state = ctypes.cast(source, ctypes.POINTER(runtime.PowerState)).contents
        self.thread_power = state.ControlMask, state.StateMask
        return not self.fail


def test_high_qos_disables_only_speed_throttling_and_restores_original_state():
    kernel = Kernel(control=3, state=3)
    policy = runtime.WindowsScheduling(kernel)
    status = policy.enable()
    assert kernel.power == (3, 2)  # Honor the unrelated timer-throttling policy.
    assert kernel.priority == runtime.ABOVE_NORMAL
    assert status["power_throttling_disabled"] and status["priority"] == "above_normal"
    policy.prepare_thread()
    assert kernel.thread_power == (3, 2)
    assert policy.status()["search_thread_high_qos"] is True
    policy.restore()
    assert kernel.power == (3, 3) and kernel.priority == 0x20
    writes = len(kernel.writes)
    policy.restore()
    assert len(kernel.writes) == writes


def test_system_managed_zero_masks_are_explicitly_overridden():
    kernel = Kernel()
    policy = runtime.WindowsScheduling(kernel)
    policy.enable()
    assert kernel.power == (1, 0)  # (0, 0) is heuristic/system mode, not explicit HighQoS.
    policy.restore()
    assert kernel.power == (0, 0)


@pytest.mark.parametrize("priority", [runtime.ABOVE_NORMAL, 0x80, 0x100])
def test_preserves_user_selected_higher_priority(priority):
    kernel = Kernel(priority=priority)
    policy = runtime.WindowsScheduling(kernel)
    policy.enable()
    policy.restore()
    assert kernel.priority == priority and kernel.priorities == []


def test_api_rejection_is_reported_without_stopping_the_server():
    kernel = Kernel(fail=True)
    policy = runtime.WindowsScheduling(kernel)
    status = policy.enable()
    assert not status["power_throttling_disabled"]
    assert status["priority"] == "normal" and len(status["notes"]) == 2
    policy.restore()
    assert kernel.power == (0, 0) and kernel.priority == 0x20


def test_unsupported_power_api_still_allows_priority_adjustment():
    kernel = Kernel()
    kernel.GetProcessInformation = kernel.SetProcessInformation = None
    policy = runtime.WindowsScheduling(kernel)
    status = policy.enable()
    assert status["priority"] == "above_normal"
    assert not status["power_throttling_disabled"]
    assert status["notes"]
    policy.restore()
    assert kernel.priority == 0x20


def test_system_mode_and_import_do_not_load_or_modify_win32(monkeypatch):
    def unexpected_api():
        raise AssertionError("System mode should not touch Win32")

    monkeypatch.setattr(runtime, "_policy", None)
    monkeypatch.setattr(runtime, "_windows_api", unexpected_api)
    assert runtime.configure_runtime("system")["mode"] == "system"
    runtime.prepare_search_thread()
    with pytest.raises(ValueError):
        runtime.configure_runtime("realtime")


def test_restoring_does_not_report_stale_enabled_state(monkeypatch):
    kernel = Kernel()
    policy = runtime.WindowsScheduling(kernel)
    policy.enable()
    monkeypatch.setattr(runtime, "_policy", policy)
    assert runtime.runtime_status()["power_throttling_disabled"]
    runtime.restore_runtime()
    assert kernel.priority == 0x20 and kernel.power == (0, 0)
    assert runtime.runtime_status()["mode"] == "system"
    assert not runtime.runtime_status()["power_throttling_disabled"]
