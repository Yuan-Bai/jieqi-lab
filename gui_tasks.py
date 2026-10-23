"""有界的后台搜索通道；Tk 线程只提交快照及轮询结果。

每个通道仅一个线程、一个待办任务。新局面取消旧搜索，并替换排队任务。
通过 RNG 的模拟采样边界协作取消，无需改变训练/搜索代码的接口。
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass


class SearchCancelled(Exception):
    pass


class SearchRandom(random.Random):
    def __init__(self):
        super().__init__()
        self.cancelled = threading.Event()

    def check(self):
        if self.cancelled.is_set():
            raise SearchCancelled()

    def random(self):
        self.check()
        return super().random()

    def getrandbits(self, k):
        self.check()
        return super().getrandbits(k)


class SearchBelief:
    """只包装根观测，确保全明子局面也在每次模拟开始时检查取消。"""
    def __init__(self, belief, rng):
        self.belief, self.rng = belief, rng

    def info_state(self, state, observer):
        self.rng.check()
        return _SearchInformation(self.belief.info_state(state, observer), self.rng)

    def __getattr__(self, name):
        return getattr(self.belief, name)


class _SearchInformation:
    def __init__(self, info, rng):
        self.info, self.rng = info, rng

    def determinize(self, rng):
        self.rng.check()
        return self.info.determinize(rng)

    def __getattr__(self, name):
        return getattr(self.info, name)


@dataclass
class TaskResult:
    token: int
    value: object = None
    elapsed: float = 0.0
    error: str | None = None


class LatestTaskRunner:
    def __init__(self):
        self._condition = threading.Condition()
        self._pending = None
        self._active_cancel = None
        self._result = None
        self._generation = 0
        self._closed = False
        self._thread = None
        self.rng = SearchRandom()

    def invalidate(self):
        with self._condition:
            self._generation += 1
            if self._active_cancel is not None:
                self._active_cancel.set()
            self._pending = None
            self._result = None
            return self._generation

    def submit(self, work):
        with self._condition:
            if self._closed:
                raise RuntimeError("搜索通道已关闭")
            self._generation += 1
            token = self._generation
            if self._active_cancel is not None:
                self._active_cancel.set()
            self._pending = (token, work, threading.Event())
            self._result = None
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
            self._condition.notify()
            return token

    def poll(self, token):
        with self._condition:
            if self._result is not None and self._result.token == token:
                result, self._result = self._result, None
                return result
        return None

    def close(self):
        with self._condition:
            self._closed = True
            self._generation += 1
            self._pending = None
            self._result = None
            if self._active_cancel is not None:
                self._active_cancel.set()
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    return
                token, work, cancel = self._pending
                self._pending = None
                self._active_cancel = cancel
            self.rng.cancelled = cancel
            start = time.perf_counter()
            try:
                self.rng.check()
                result = TaskResult(token, work(self.rng), time.perf_counter() - start)
            except SearchCancelled:
                result = None
            except Exception as exc:
                result = TaskResult(token, error=f"{type(exc).__name__}: {exc}")
            with self._condition:
                self._active_cancel = None
                if result is not None and token == self._generation and not self._closed:
                    self._result = result
