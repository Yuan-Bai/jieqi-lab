"""Synthetic SQLite fixtures only; no Nsight/CUDA process is launched by these tests."""
import sqlite3
import unittest

from tools.inspect_native_runtime import inspect, decode_pid


class NativeRuntimeCoverageTests(unittest.TestCase):
    def fixture(self, mode="single"):
        connection = sqlite3.connect(":memory:")
        self.addCleanup(connection.close)
        connection.executescript("""
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER, copyKind INTEGER, bytes INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
            CREATE TABLE DIAGNOSTIC_EVENT (severity INTEGER, text TEXT);
        """)
        root = 10
        pids = [root] if mode == "single" else [11, 12]
        gid = lambda pid: (1 << 48) + (pid << 24)
        connection.execute("INSERT INTO NVTX_EVENTS VALUES (100, 1000, ?, 'jieqi_native_window', NULL)", (gid(root) + 5,))
        for pid in pids:
            for index in range(8):
                start = 120 + index * 50
                connection.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, 'jieqi/compat/forward', NULL)", (start, start + 40, gid(pid) + 5))
                for offset in (1, 2):
                    correlation = index * 2 + offset
                    connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)", (start + 25, start + 30, gid(pid), correlation))
                    connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)", (start + 5, start + 10, gid(pid) + 5, correlation))
            connection.execute("INSERT INTO NVTX_EVENTS VALUES (600, 900, ?, 'jieqi/compat/readback', NULL)", (gid(pid) + 5,))
            for correlation, size in ((17, 65536), (18, 14400)):
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (650, 700, ?, ?, 2, ?)", (gid(pid), correlation, size))
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (620, 710, ?, ?)", (gid(pid) + 5, correlation))
        workload = dict(kind="native_runtime_probe", mode=mode, pid=root, models_loaded=False, training_started=False,
                        rows=[dict(pid=pid, thread_id=5, exact=True, dtype="torch.float32", tf32_matmul=False, tf32_cudnn=True) for pid in pids])
        return connection, workload

    def test_single_and_spawn(self):
        for mode in ("single", "spawn"):
            connection, workload = self.fixture(mode)
            result = inspect(connection, workload)
            self.assertTrue(result["known_workload_coverage_passed"], result["errors"])
            self.assertFalse(result["lossless_trace_verified"])

    def test_warnings_are_preserved_not_silenced(self):
        connection, workload = self.fixture()
        connection.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (2, 'Not all CUDA events might have been collected.')")
        result = inspect(connection, workload)
        self.assertTrue(result["known_workload_coverage_passed"])
        self.assertEqual(len(result["warnings"]), 1)
        self.assertFalse(result["lossless_trace_verified"])

    def test_missing_kernel_rejected(self):
        connection, workload = self.fixture()
        connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=1")
        self.assertFalse(inspect(connection, workload)["known_workload_coverage_passed"])

    def test_missing_child_rejected(self):
        connection, workload = self.fixture("spawn")
        connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=?", ((1 << 48) + (12 << 24),))
        self.assertFalse(inspect(connection, workload)["known_workload_coverage_passed"])

    def test_open_window_rejected(self):
        connection, workload = self.fixture()
        connection.execute("UPDATE NVTX_EVENTS SET end=NULL WHERE text='jieqi_native_window'")
        self.assertFalse(inspect(connection, workload)["known_workload_coverage_passed"])

    def test_outside_window_rejected(self):
        connection, workload = self.fixture()
        connection.execute("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET end=1001")
        self.assertFalse(inspect(connection, workload)["known_workload_coverage_passed"])

    def test_uncorrelated_kernel_rejected(self):
        connection, workload = self.fixture()
        connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=1")
        self.assertFalse(inspect(connection, workload)["known_workload_coverage_passed"])

    def test_pid_layout(self):
        self.assertEqual(decode_pid((1 << 48) + (6080 << 24) + 13760), 6080)

    def test_lifecycle_retains_but_excludes_warmup(self):
        connection, workload = self.fixture("spawn")
        gid = (1 << 48) + (11 << 24)
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10, 20, ?, 99)", (gid,))
        connection.execute("INSERT INTO NVTX_EVENTS VALUES (10, 20, ?, 'jieqi/compat/readback', NULL)", (gid + 5,))
        self.assertFalse(inspect(connection, workload)["known_workload_coverage_passed"])
        result = inspect(connection, workload, "lifecycle")
        self.assertTrue(result["known_workload_coverage_passed"], result["errors"])
        self.assertEqual(result["total_counts"]["kernels"], 33)
        self.assertEqual(result["counts"]["kernels"], 32)

    def test_lifecycle_rejects_crossing_event(self):
        connection, workload = self.fixture()
        connection.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=50 WHERE correlationId=1")
        self.assertFalse(inspect(connection, workload, "lifecycle")["known_workload_coverage_passed"])

    def cpu_fixture(self):
        connection, workload = self.fixture("spawn")
        connection.executescript("""
            DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL;
            DELETE FROM CUPTI_ACTIVITY_KIND_MEMCPY;
            DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME;
            INSERT INTO StringIds VALUES (99,'python311.dll');
            CREATE TABLE COMPOSITE_EVENTS (id INTEGER, start INTEGER, globalTid INTEGER, threadState INTEGER, cpuCycles INTEGER);
            CREATE TABLE SAMPLING_CALLCHAINS (id INTEGER, stackDepth INTEGER, originalIP INTEGER, unresolved INTEGER, kernelMode INTEGER, module INTEGER);
            CREATE TABLE SCHED_EVENTS (globalTid INTEGER, start INTEGER);
        """)
        for index, row in enumerate(workload["rows"]):
            tid = (1 << 48) + (row["pid"] << 24) + 5
            connection.execute("INSERT INTO COMPOSITE_EVENTS VALUES (?,300,?,1,0)", (index, tid))
            connection.execute("INSERT INTO SAMPLING_CALLCHAINS VALUES (?,0,1234,1,NULL,99)", (index,))
            connection.execute("INSERT INTO SCHED_EVENTS VALUES (?,300)", (tid,))
        return connection, workload

    def test_cpu_known_coverage_without_pmu_or_symbols(self):
        connection, workload = self.cpu_fixture()
        result = inspect(connection, workload, "lifecycle", "cpu")
        self.assertTrue(result["known_workload_coverage_passed"], result["errors"])
        for row in result["cpu"]["per_pid"]:
            self.assertEqual(row["window_stack_events"], 1)
            self.assertEqual(row["positive_cpu_cycle_events"], 0)
            self.assertEqual(row["unknown_kernel_mode_rows"], 1)
            self.assertEqual(row["unresolved_frame_rows"], 1)
        self.assertFalse(result["lossless_trace_verified"])

    def test_cpu_other_thread_does_not_satisfy_gate(self):
        connection, workload = self.cpu_fixture()
        connection.execute("UPDATE COMPOSITE_EVENTS SET globalTid=globalTid+1")
        self.assertFalse(inspect(connection, workload, "lifecycle", "cpu")["known_workload_coverage_passed"])

    def test_cpu_missing_stack_rejected(self):
        connection, workload = self.cpu_fixture()
        connection.execute("DELETE FROM SAMPLING_CALLCHAINS WHERE id=0")
        self.assertFalse(inspect(connection, workload, "lifecycle", "cpu")["known_workload_coverage_passed"])

    def test_cpu_warmup_samples_do_not_satisfy_gate(self):
        connection, workload = self.cpu_fixture()
        connection.execute("UPDATE COMPOSITE_EVENTS SET start=50")
        self.assertFalse(inspect(connection, workload, "lifecycle", "cpu")["known_workload_coverage_passed"])

    def test_cpu_missing_sched_rejected(self):
        connection, workload = self.cpu_fixture()
        connection.execute("DELETE FROM SCHED_EVENTS")
        self.assertFalse(inspect(connection, workload, "lifecycle", "cpu")["known_workload_coverage_passed"])

    def test_cpu_collector_error_rejected_but_verbose_is_not_error(self):
        connection, workload = self.cpu_fixture()
        connection.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (4,'Verbose diagnostic')")
        self.assertTrue(inspect(connection, workload, "lifecycle", "cpu")["known_workload_coverage_passed"])
        connection.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (3,'Perf AccessDenied')")
        result = inspect(connection, workload, "lifecycle", "cpu")
        self.assertFalse(result["known_workload_coverage_passed"])
        self.assertEqual(len(result["diagnostic_errors"]), 1)


if __name__ == "__main__":
    unittest.main()
