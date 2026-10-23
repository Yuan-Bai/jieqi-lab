"""Native trace analysis uses in-memory fixtures, never GPU or production data."""
import sqlite3
import unittest

from tools.inspect_native_selfplay import inspect, intersect_ns, interval_union, scheduling_segments


class NativeSelfplayInspectionTests(unittest.TestCase):
    def fixture(self):
        c = sqlite3.connect(":memory:")
        self.addCleanup(c.close)
        c.executescript("""
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES (1,'cudaStreamSynchronize'),(2,'synthetic_kernel'),(3,'cudaLaunchKernel_v7000'),(4,'cudaMemcpyAsync_v3020');
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, globalPid INTEGER, demangledName INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, globalPid INTEGER, copyKind INTEGER, bytes INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER, nameId INTEGER, correlationId INTEGER);
            CREATE TABLE DIAGNOSTIC_EVENT (severity INTEGER, text TEXT);
        """)
        gid = lambda pid: (1 << 48) + (pid << 24)
        c.execute("INSERT INTO NVTX_EVENTS VALUES (100,1000,?,'jieqi_native_window',NULL)", (gid(10) + 1,))
        rows = []
        for pid in range(11, 19):
            for index in range(18):
                strong = index % 2 == 0
                rows.append(dict(pid=pid, native_thread_id=5, strong=strong))
                c.execute("INSERT INTO NVTX_EVENTS VALUES (120,900,?,?,NULL)", (gid(pid)+5, "jieqi/root/3200" if strong else "jieqi/root/800"))
            for phase in ("evaluation", "encode", "h2d", "forward", "softmax", "policy_d2h", "value_d2h"):
                c.execute("INSERT INTO NVTX_EVENTS VALUES (200,800,?,?,NULL)", (gid(pid)+5, "jieqi/"+phase))
            c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (300,400,?,2,1)", (gid(pid),))
            c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (500,550,?,2,1024,2)", (gid(pid),))
            c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (450,600,?,1,0)", (gid(pid)+5,))
            c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (290,310,?,3,1)", (gid(pid)+5,))
            c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (450,500,?,4,2)", (gid(pid)+5,))
        workload = dict(kind="native_profile_capture", pid=10, paired_roots_exact=True, production_unchanged=True,
                        groups=[dict(rows=rows[index:index+36]) for index in range(0, 144, 36)])
        return c, workload

    def test_complete_known_labels_not_losslessness(self):
        c, workload = self.fixture()
        result = inspect(c, workload)
        self.assertTrue(result["known_root_label_coverage_passed"], result["errors"])
        self.assertFalse(result["gpu_event_completeness_verified"])
        self.assertFalse(result["lossless_trace_verified"])
        self.assertAlmostEqual(result["gpu_kernel_copy_union_fraction"], 150/900)

    def test_missing_root_rejected(self):
        c, workload = self.fixture()
        c.execute("DELETE FROM NVTX_EVENTS WHERE rowid=(SELECT min(rowid) FROM NVTX_EVENTS WHERE text='jieqi/root/800')")
        self.assertFalse(inspect(c, workload)["known_root_label_coverage_passed"])

    def test_missing_phase_rejected(self):
        c, workload = self.fixture()
        c.execute("DELETE FROM NVTX_EVENTS WHERE text='jieqi/policy_d2h'")
        self.assertFalse(inspect(c, workload)["known_root_label_coverage_passed"])

    def test_child_gpu_missing_rejected(self):
        c, workload = self.fixture()
        c.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=?", ((1 << 48)+(11 << 24),))
        self.assertFalse(inspect(c, workload)["known_root_label_coverage_passed"])

    def test_crossing_window_rejected(self):
        c, workload = self.fixture()
        c.execute("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET end=1001")
        self.assertFalse(inspect(c, workload)["known_root_label_coverage_passed"])

    def test_warnings_preserved(self):
        c, workload = self.fixture()
        c.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (2,'Not all events collected')")
        self.assertEqual(len(inspect(c, workload)["warnings"]), 1)

    def test_non_cuda_collector_error_kept_separate(self):
        c, workload = self.fixture()
        c.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (3,'Perf: AccessDeniedException')")
        result = inspect(c, workload)
        self.assertTrue(result["known_root_label_coverage_passed"])
        self.assertEqual(len(result["diagnostic_errors"]), 1)
        self.assertFalse(result["lossless_trace_verified"])

    def test_verbose_diagnostic_is_not_error(self):
        c, workload = self.fixture()
        c.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (4,'Verbose message')")
        result = inspect(c, workload)
        self.assertEqual(result["diagnostic_errors"], [])
        self.assertEqual(result["warnings"], [])

    def test_pid_and_correlation_must_both_match(self):
        c, workload = self.fixture()
        c.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId=99")
        self.assertFalse(inspect(c, workload)["known_root_label_coverage_passed"])

    def test_interval_union(self):
        self.assertEqual(interval_union([(0, 10), (1, 2), (3, 20), (20, 30), (40, 50)]), 40)
        self.assertEqual(interval_union([]), 0)
        with self.assertRaises(ValueError):
            interval_union([(2, 1)])

    def cpu_fixture(self):
        c, workload = self.fixture()
        c.executescript("""
            DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL;
            DROP TABLE CUPTI_ACTIVITY_KIND_MEMCPY;
            DROP TABLE CUPTI_ACTIVITY_KIND_RUNTIME;
            DELETE FROM NVTX_EVENTS WHERE text!='jieqi_native_window';
            INSERT INTO StringIds VALUES (5,'python311.dll'),(6,'nvcuda64.dll');
            CREATE TABLE COMPOSITE_EVENTS (id INTEGER PRIMARY KEY, start INTEGER, globalTid INTEGER, cpuCycles INTEGER);
            CREATE TABLE SAMPLING_CALLCHAINS (id INTEGER, stackDepth INTEGER, module INTEGER, unresolved INTEGER, kernelMode INTEGER, originalIP INTEGER);
            CREATE TABLE SCHED_EVENTS (start INTEGER, globalTid INTEGER, isSchedIn INTEGER, threadState INTEGER, threadBlock INTEGER);
            CREATE TABLE ENUM_SAMPLING_THREAD_STATE (id INTEGER, name TEXT);
            CREATE TABLE ENUM_SCHEDULING_THREAD_BLOCK (id INTEGER, name TEXT);
            INSERT INTO ENUM_SAMPLING_THREAD_STATE VALUES (1,'Running'),(7,'Waiting');
            INSERT INTO ENUM_SCHEDULING_THREAD_BLOCK VALUES (0,'NonBlocked'),(25,'UserRequest');
        """)
        for pid in range(11, 19):
            tid = (1 << 48) + (pid << 24) + 5
            for index in range(18):
                c.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)",
                          (130+40*index, 165+40*index, tid, "jieqi/root/3200" if index % 2 == 0 else "jieqi/root/800"))
            c.execute("INSERT INTO NVTX_EVENTS VALUES (135,168,?,'jieqi/evaluation',NULL)", (tid,))
            # Extend the first root to contain the complete evaluation range.
            c.execute("UPDATE NVTX_EVENTS SET end=169 WHERE start=130 AND globalTid=?", (tid,))
            for index, phase in enumerate(("encode", "h2d", "forward", "softmax", "policy_d2h", "value_d2h")):
                c.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)", (140+4*index, 144+4*index, tid, "jieqi/"+phase))
            for start, active, state, block in ((90,1,1,None), (280,0,7,25), (290,1,1,None), (950,0,1,0), (1050,1,1,None)):
                c.execute("INSERT INTO SCHED_EVENTS VALUES (?,?,?,?,?)", (start,tid,active,state,block))
            for index, (timestamp, cycles, module) in enumerate(((150,1,5), (158,1,6), (158,0,6), (80,1,6))):
                event_id = pid*100+index
                c.execute("INSERT INTO COMPOSITE_EVENTS VALUES (?,?,?,?)", (event_id,timestamp,tid,cycles))
                c.execute("INSERT INTO SAMPLING_CALLCHAINS VALUES (?,0,?,1,NULL,123)", (event_id,module))
                c.execute("INSERT INTO SAMPLING_CALLCHAINS VALUES (?,1,6,1,NULL,456)", (event_id,))
        return c, workload

    def test_cpu_only_periodic_leaf_samples_count(self):
        c, workload = self.cpu_fixture()
        result = inspect(c, workload, "cpu")
        self.assertTrue(result["known_cpu_coverage_passed"], result["errors"])
        self.assertEqual(result["periodic_root_samples"], 16)
        self.assertEqual(result["periodic_phases"]["policy_d2h"]["samples"], 8)
        self.assertEqual(result["periodic_leaf_modules"], [dict(module="python311.dll", samples=8), dict(module="nvcuda64.dll", samples=8)])
        self.assertFalse(result["lossless_trace_verified"])
        root = result["phase_scheduling"]["root"]
        self.assertEqual(root["wall_ns"], 8*(18*35+4))
        self.assertEqual(root["scheduled_off_ns"], 8*5)
        self.assertEqual(root["unaccounted_ns"], 0)

    def test_cpu_other_thread_not_coverage(self):
        c, workload = self.cpu_fixture()
        c.execute("UPDATE COMPOSITE_EVENTS SET globalTid=globalTid+1 WHERE globalTid=?", ((1<<48)+(11<<24)+5,))
        self.assertFalse(inspect(c, workload, "cpu")["known_cpu_coverage_passed"])

    def test_cpu_missing_periodic_leaf_rejected(self):
        c, workload = self.cpu_fixture()
        c.execute("DELETE FROM SAMPLING_CALLCHAINS WHERE stackDepth=0")
        self.assertFalse(inspect(c, workload, "cpu")["known_cpu_coverage_passed"])

    def test_cpu_context_switch_stacks_do_not_replace_samples(self):
        c, workload = self.cpu_fixture()
        c.execute("UPDATE COMPOSITE_EVENTS SET cpuCycles=0")
        result = inspect(c, workload, "cpu")
        self.assertEqual(result["periodic_root_samples"], 0)
        self.assertFalse(result["known_cpu_coverage_passed"])

    def test_cpu_missing_phase_and_boundary_schedule_rejected(self):
        c, workload = self.cpu_fixture()
        c.execute("DELETE FROM NVTX_EVENTS WHERE text='jieqi/h2d'")
        self.assertFalse(inspect(c, workload, "cpu")["known_cpu_coverage_passed"])
        c, workload = self.cpu_fixture()
        c.execute("DELETE FROM SCHED_EVENTS WHERE start=90")
        self.assertFalse(inspect(c, workload, "cpu")["known_cpu_coverage_passed"])

    def test_cpu_overlapping_subphases_rejected_without_samples_in_overlap(self):
        c, workload = self.cpu_fixture()
        c.execute("UPDATE NVTX_EVENTS SET start=143 WHERE text='jieqi/h2d'")
        self.assertFalse(inspect(c, workload, "cpu")["known_cpu_coverage_passed"])

    def test_cpu_error_rejected_verbose_kept(self):
        c, workload = self.cpu_fixture()
        c.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (4,'Verbose')")
        self.assertTrue(inspect(c, workload, "cpu")["known_cpu_coverage_passed"])
        c.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (3,'Collector failed')")
        self.assertFalse(inspect(c, workload, "cpu")["known_cpu_coverage_passed"])

    def test_cpu_window_half_open_and_non_target_ignored(self):
        c, workload = self.cpu_fixture()
        c.execute("INSERT INTO COMPOSITE_EVENTS VALUES (9999,1000,?,1)", ((1<<48)+(11<<24)+5,))
        c.execute("INSERT INTO COMPOSITE_EVENTS VALUES (10000,150,999,1)")
        self.assertEqual(inspect(c, workload, "cpu")["periodic_root_samples"], 16)

    def test_schedule_clips_boundaries_and_rejects_duplicate_in(self):
        events = [dict(start=start, isSchedIn=active, threadState=1, threadBlock=0)
                  for start, active in ((90,1),(150,0),(180,0),(200,1),(220,1),(300,0))]
        spans, anomalies = scheduling_segments(events, (100,250))
        self.assertEqual(anomalies, 1)
        self.assertEqual(intersect_ns([(left,right) for left,right,on,_,_ in spans if on], [(110,240)]), 60)
        self.assertEqual(intersect_ns([], [(110,240)]), 0)


if __name__ == "__main__":
    unittest.main()
