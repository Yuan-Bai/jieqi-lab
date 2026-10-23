import unittest

from tools import profile_python_selfplay as sampler


class SamplingTests(unittest.TestCase):
    def example(self):
        frames = [dict(file=str(sampler.THIS), name=name, line=1)
                  for name in ("measured_task", "measured_ordinary", "warm_task", "compatibility_hot_loop")]
        frames += [dict(file=str(sampler.ROOT/"azero.py"), name="search", line=1750),
                   dict(file=str(sampler.ROOT/"jieqi.py"), name="apply_move", line=500)]
        profile = dict(name='Process 11 Thread 22 "MainThread"', type="sampled", unit="seconds",
                       samples=[[0, 1, 4, 5, 5], [2, 4, 5], [0, 1, 5]], weights=[0.01]*3)
        return dict(exporter="py-spy@0.4.2", shared=dict(frames=frames), profiles=[profile])

    def analyze(self, data, **kwargs):
        return sampler.analyze_speedscope(data, [dict(pid=11, native_thread_id=22)],
                                         minimum_total=1, minimum_worker=1, **kwargs)

    def test_search_marker_and_no_recursive_double_count(self):
        result = self.analyze(self.example())
        self.assertEqual(result["eligible_samples"], 1)
        self.assertEqual(result["excluded_samples"], 2)
        self.assertEqual(result["exclusive"][0]["name"], "apply_move")
        self.assertEqual(result["exclusive"][0]["fraction"], 1)
        self.assertTrue(all(row["samples"] == 1 for row in result["inclusive"]))
        self.assertEqual(result["budgets"]["800"]["samples"], 1)

    def test_other_threads_and_parents_excluded(self):
        data = self.example()
        extra = dict(data["profiles"][0], name='Process 11 Thread 33 "Other"')
        data["profiles"].append(extra)
        result = self.analyze(data)
        self.assertEqual(result["eligible_samples"], 1)
        self.assertEqual(result["excluded_samples"], 5)

    def test_coverage_missing_worker_fails(self):
        result = sampler.analyze_speedscope(self.example(),
            [dict(pid=11, native_thread_id=22), dict(pid=12, native_thread_id=23)],
            minimum_worker=1, minimum_total=1)
        self.assertFalse(result["coverage_gate_passed"])

    def test_bad_exporter_or_indices_or_weights_rejected(self):
        for mutate in (lambda d: d.update(exporter="unknown"),
                       lambda d: d["profiles"][0]["samples"].append([99]),
                       lambda d: d["profiles"][0].update(weights=[-1, 0.01, 0.01])):
            data = self.example()
            mutate(data)
            with self.assertRaises(ValueError):
                self.analyze(data)

    def test_wrong_profile_name_fails_closed(self):
        data = self.example()
        data["profiles"][0]["name"] = "Thread 22"
        with self.assertRaisesRegex(ValueError, "no eligible"):
            self.analyze(data)

    def test_compatibility_separate_from_search(self):
        data = self.example()
        data["profiles"][0].update(samples=[[3]], weights=[0.01])
        result = self.analyze(data, compatibility=True)
        self.assertTrue(result["coverage_gate_passed"])
        self.assertEqual(result["budgets"], {})

    def test_strong_marker_and_windows_path_case(self):
        data = self.example()
        data["shared"]["frames"][1]["name"] = "measured_strong"
        data["shared"]["frames"][0]["file"] = str(sampler.THIS).upper().replace("\\", "/")
        self.assertEqual(self.analyze(data)["budgets"]["3200"]["samples"], 1)

    def test_weights_not_used_as_cpu_time(self):
        data = self.example()
        data["profiles"][0]["weights"] = [17, 17, 17]
        result = self.analyze(data)
        self.assertEqual(result["eligible_samples"], 1)
        self.assertNotIn("cpu_seconds", result)
        self.assertIn("NOT CPU seconds", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
