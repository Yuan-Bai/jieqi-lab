"""Windows Ctrl+C probe: isolated hidden console, no training or model writes.

Use --scenario startup to interrupt a spawn while the parent serializes a large payload
and the child imports torch; active interrupts a running worker. Artifacts go to a new path.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIXTURE = '''
import os, sys, time, signal, threading, importlib.abc, importlib.machinery
from pathlib import Path
sys.path.insert(0, {project_root!r})
root = Path({output!r})
if __name__ == "__main__" and os.name == "nt":
    import ctypes
    ctypes.WinDLL("kernel32", use_last_error=True).SetConsoleCtrlHandler(None, False)
try:
    import training_runtime as runtime
except ModuleNotFoundError:
    runtime = None
if runtime:
    runtime.bootstrap(__name__)
if __name__ == "__mp_main__" and {scenario!r} == "startup":
    class SlowTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "torch":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            actual = spec.loader
            class Loader(importlib.abc.Loader):
                def create_module(self, spec):
                    return actual.create_module(spec)
                def exec_module(self, module):
                    (root / "child_ready.json").write_text(str(os.getpid()))
                    time.sleep(30)
                    actual.exec_module(module)
            spec.loader = Loader()
            return spec
    sys.meta_path.insert(0, SlowTorch())
import azero as az
def work(payload):
    (root / "child_ready.json").write_text(str(os.getpid()))
    time.sleep(30)
if __name__ == "__main__":
    az.install_interrupt_handler()
    p = az.mp.get_context("spawn").Process(target=work, args=(bytes(32 * 1024 * 1024),))
    p.start()
    (root / "start_returned.json").write_text("yes")
    if {scenario!r} == "stderr_locked":
        class LockedStderr:
            def write(self, text):
                time.sleep(30)
            def flush(self):
                time.sleep(30)
        sys.stderr = LockedStderr()
    p.join()
'''

CONTROLLER = '''
import ctypes, sys
from ctypes import wintypes
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.AttachConsole.argtypes = [wintypes.DWORD]
k.AttachConsole.restype = wintypes.BOOL
k.SetConsoleCtrlHandler.argtypes = [ctypes.c_void_p, wintypes.BOOL]
k.SetConsoleCtrlHandler.restype = wintypes.BOOL
k.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
k.GenerateConsoleCtrlEvent.restype = wintypes.BOOL
k.FreeConsole()
if not k.AttachConsole(int(sys.argv[1])):
    raise ctypes.WinError(ctypes.get_last_error())
if not k.SetConsoleCtrlHandler(None, True):
    raise ctypes.WinError(ctypes.get_last_error())
ignore_event = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)(lambda event: event in (0, 1))
if not k.SetConsoleCtrlHandler(ignore_event, True):
    raise ctypes.WinError(ctypes.get_last_error())
if not k.GenerateConsoleCtrlEvent(int(sys.argv[2]), 0):
    raise ctypes.WinError(ctypes.get_last_error())
# Do not exit/detach before the console has dispatched the event.
import time; time.sleep(0.2)
'''


def run_probe(output: Path, scenario="startup", event=0, timeout=5.0) -> dict:
    if os.name != "nt":
        raise RuntimeError("Windows console probe requires Windows")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "fixture.py"
    fixture.write_text(FIXTURE.format(project_root=str(PROJECT_ROOT), output=str(output), scenario=scenario), encoding="utf-8")
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0  # Hidden dedicated console, never signal the user's terminal.
    import psutil  # Probe-only optional dependency; the runtime uses only the standard library.
    with (output / "stdout.txt").open("wb") as stdout, (output / "stderr.txt").open("wb") as stderr:
        process = subprocess.Popen([sys.executable, str(fixture)], cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
        owned = psutil.Process(process.pid)
        children = []
        try:
            deadline = time.monotonic() + 30
            while not (output / "child_ready.json").exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("probe worker did not reach the expected phase")
                time.sleep(0.05)
            if scenario != "startup":
                while not (output / "start_returned.json").exists():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("probe parent did not finish spawning")
                    time.sleep(0.05)
            child_pid = int((output / "child_ready.json").read_text())
            children = [p for p in owned.children(recursive=True) if p.name().lower() != "conhost.exe"]
            blocked_start = not (output / "start_returned.json").exists()
            start = time.monotonic()
            if event == 2:
                process.terminate()  # External stop must also release the job and its children.
            else:
                controller = subprocess.run([sys.executable, "-c", CONTROLLER, str(process.pid), str(event)],
                                            creationflags=subprocess.CREATE_NO_WINDOW, timeout=5, capture_output=True)
                if controller.returncode:
                    raise RuntimeError(controller.stderr.decode("utf-8", errors="replace"))
            try:
                process.wait(timeout=timeout)
                exited = True
            except subprocess.TimeoutExpired:
                exited = False
            if exited:
                psutil.wait_procs(children, timeout=1)
            elapsed = time.monotonic() - start
            survivors = [p.pid for p in children if p.is_running()]
            result = dict(scenario=scenario, event=event, parent_pid=process.pid, child_pid=child_pid,
                          blocked_during_spawn=blocked_start, exited=exited, exit_code=process.returncode,
                          elapsed_seconds=elapsed, surviving_children=survivors)
        finally:
            # Only this newly created fixture's tree; failures must not leak sleepers.
            try:
                children += [p for p in owned.children(recursive=True) if p.name().lower() != "conhost.exe"]
            except psutil.NoSuchProcess:
                pass
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            psutil.wait_procs(children, timeout=5)
    result["keyboard_interrupt_traceback"] = "KeyboardInterrupt" in (output / "stderr.txt").read_text(encoding="utf-8", errors="replace")
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--scenario", choices=("startup", "active", "stderr_locked"), default="startup")
    parser.add_argument("--event", choices=("ctrl-c", "ctrl-break", "terminate"), default="ctrl-c")
    parser.add_argument("--timeout", type=float, default=5)
    args = parser.parse_args(argv)
    event = {"ctrl-c": 0, "ctrl-break": 1, "terminate": 2}[args.event]
    result = run_probe(args.out, args.scenario, event, args.timeout)
    print(json.dumps(result))
    expected_exit = 1 if event == 2 else 130
    return 0 if result["exited"] and result["exit_code"] == expected_exit and not result["surviving_children"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
