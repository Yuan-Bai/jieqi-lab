# cython: language_level=3, infer_types=False, annotation_typing=False, boundscheck=True, wraparound=True
"""Isolated immutable-board identity memo; no mutable-state or network cache."""
import sys

API_VERSION = 1
SOURCE_SHA256 = NATIVE_SOURCE_SHA256
CORE_SHA256 = NATIVE_CORE_SHA256
CODE_SHA256 = NATIVE_CODE_SHA256
MODULE = ORIGINAL = SIGNATURE = DARK = INFO_CLASS = None


def initialize(module, fingerprint):
    global MODULE, ORIGINAL, SIGNATURE, DARK, INFO_CLASS
    if fingerprint != CODE_SHA256:
        raise ImportError("experimental repetition function fingerprint mismatch")
    if MODULE is not None:
        if MODULE is not module:
            raise ImportError("experimental repetition memo cannot serve another module")
        return
    MODULE = module
    ORIGINAL = module._repetition_entries
    SIGNATURE = module._public_material_signature
    DARK = module.jq.is_dark
    INFO_CLASS = module.im.InformationState


cdef bint standard_board(object board):
    cdef object piece
    if type(board) is not tuple or len(board) != 90:
        return False
    for piece in board:
        if type(piece) is not int or piece < 0 or piece > 31:
            return False
    return True


cdef class RepetitionMemo:
    cdef dict _boards
    cdef int _capacity
    cdef bint _enabled, _collect
    cdef unsigned long long _calls, _fast, _fallback, _queries, _hits, _misses, _evictions, _bytes

    def __init__(self, *, capacity=4096, enabled=True, collect_stats=False):
        if MODULE is None:
            raise RuntimeError("initialize experimental repetition memo first")
        if type(capacity) is not int or capacity < 1 or capacity > 4096:
            raise ValueError("capacity must be an integer in [1, 4096]")
        self._capacity, self._enabled, self._collect = capacity, bool(enabled), bool(collect_stats)
        self._boards = {}

    def clear(self):
        self._boards.clear()
        self._calls = self._fast = self._fallback = self._queries = 0
        self._hits = self._misses = self._evictions = self._bytes = 0

    def supports(self, info):
        cdef object namespace = MODULE.__dict__
        return (type(info) is INFO_CLASS and type(info.board) is list
                and type(info.side) is int and type(info.position_counts) is dict
                and type(info.repetition_history_complete) is bool
                and info.repetition_history_complete
                and MODULE._public_material_signature is SIGNATURE
                and MODULE.jq.is_dark is DARK
                and (MODULE.BOARD_SIZE, MODULE.jq.DARK_SHIFT) == (90, 4)
                and "int" not in namespace and "tuple" not in namespace
                and "bytes" not in namespace and "len" not in namespace and "sum" not in namespace)

    cdef object payload(self, object board):
        cdef object key, cached, signature
        if self._collect:
            self._queries += 1
        key = id(board)
        cached = self._boards.get(key) if self._enabled else None
        # A strong reference plus exact immutable element types makes id reuse harmless.
        if cached is not None and cached[0] is board:
            if self._collect:
                self._hits += 1
            return cached
        if self._collect:
            self._misses += 1
        if not standard_board(board):
            return None
        signature = SIGNATURE(board)
        cached = [board, signature[0], signature[1], None]
        if self._enabled:
            if len(self._boards) == self._capacity:
                self._boards.pop(next(iter(self._boards)))
                if self._collect:
                    self._evictions += 1
            self._boards[key] = cached
        return cached

    cdef object fallback(self, object info):
        if self._collect:
            self._fallback += 1
        return ORIGINAL(info)

    def entries(self, info):
        cdef object current, occupied, dark, entries, key, side, board, count, cached
        if self._collect:
            self._calls += 1
        if not self.supports(info):
            # Dependency overrides may change the original signature cache too.
            self._boards.clear()
            return self.fallback(info)
        current = tuple(info.board)
        if not standard_board(current):
            return self.fallback(info)
        occupied, dark = SIGNATURE(current)
        entries = []
        for key, count in info.position_counts.items():
            if (type(key) is not tuple or len(key) != 2 or type(key[0]) is not int
                    or type(count) is not int):
                return self.fallback(info)
            side, board = key
            cached = self.payload(board)
            if cached is None:
                return self.fallback(info)
            if cached[1] > occupied or cached[2] > dark:
                continue
            if cached[3] is None:
                cached[3] = bytes(board)
                if self._collect:
                    self._bytes += 1
            entries.append((int(side), cached[3], int(count)))
        if (info.side, current) not in info.position_counts:
            entries.append((info.side, bytes(current), 1))
        if len(entries) > MODULE.REPETITION_HISTORY_SIZE:
            raise ValueError(f"repetition history exceeds {MODULE.REPETITION_HISTORY_SIZE} positions: {len(entries)}")
        if self._collect:
            self._fast += 1
        return tuple(entries)

    def stats(self):
        cdef object retained = (sys.getsizeof(self._boards)
            + sum(sys.getsizeof(key) + sys.getsizeof(entry) + sys.getsizeof(entry[0])
                  + (sys.getsizeof(entry[3]) if entry[3] is not None else 0)
                  for key, entry in self._boards.items()))
        return dict(calls=self._calls, fast_calls=self._fast, fallback_calls=self._fallback,
                    board_queries=self._queries, hits=self._hits, misses=self._misses,
                    evictions=self._evictions, bytes_created=self._bytes, size=len(self._boards),
                    capacity=self._capacity, enabled=self._enabled, collect_stats=self._collect,
                    retained_object_bytes_upper_estimate=retained)
