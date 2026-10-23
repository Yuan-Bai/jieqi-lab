"""Explicit isolated selector build tests; no production backend switch."""
import copy
import math
import os
from pathlib import Path
import pickle
import random
import unittest
from unittest import mock

import numpy as np
import azero as az
from tools import benchmark_exact_selector as experiment


class SelectorSnapshotTests(unittest.TestCase):
    def test_snapshot_preserves_order_values_and_chance_flags(self):
        node = az.PUCTNode(1., 0)
        for index in (3, 2, 1):
            child = az.PUCTNode(index/7, index % 2, move=(index, index+1), is_chance=index == 2)
            child.visits, child.value_sum = index*2, -index/3
            node.children[child.move] = child
        snapshot = experiment.selection_snapshot(node, 1.5)
        restored, c_puct = experiment.restore_snapshot(snapshot)
        self.assertEqual(experiment.selection_snapshot(restored, c_puct), snapshot)
        self.assertEqual(experiment.ORIGINAL(node, c_puct)[0], experiment.ORIGINAL(restored, c_puct)[0])


class CompiledSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get("JIEQI_EXPERIMENT_SELECTOR_DIR")
        if not directory:
            raise unittest.SkipTest("set JIEQI_EXPERIMENT_SELECTOR_DIR to the isolated build")
        cls.selector = experiment.load_selector(Path(directory))

    def node(self, stats):
        root = az.PUCTNode(1., 0)
        for index, (prior, visits, value_sum) in enumerate(stats):
            child = az.PUCTNode(prior, 1, move=(index, index+1), is_chance=index % 2 == 0)
            child.visits, child.value_sum = visits, value_sum
            root.children[child.move] = child
        return root

    def compare(self, node, c_puct=1.5):
        before = pickle.dumps(node, protocol=5)
        outcomes = []
        for function in (experiment.ORIGINAL, self.selector.select_child):
            try:
                result = function(node, c_puct)
                outcomes.append((result, None))
            except Exception as error:
                outcomes.append((type(error), error.args))
            self.assertEqual(pickle.dumps(node, protocol=5), before)
        self.assertEqual(outcomes[0], outcomes[1])
        if outcomes[0][1] is None:
            self.assertIs(outcomes[0][0][1], outcomes[1][0][1])
            return outcomes[0][0]
        return outcomes[0]

    def test_empty_one_child_and_zero_visits(self):
        self.assertEqual(self.compare(self.node([])), (None, None))
        self.assertEqual(self.compare(self.node([(1., 0, 999.)]))[0], (0, 1))
        self.compare(self.node([(0., 0, 0.), (.7, 0, 0.), (.3, 1, -.2)]))

    def test_ties_and_insertion_order(self):
        root = self.node([(.5, 10, 0.), (.5, 10, 0.)])
        self.assertEqual(self.compare(root)[0], (0, 1))
        root.children = dict(reversed(list(root.children.items())))
        self.assertEqual(self.compare(root)[0], (1, 2))

    def test_float_neighbors_signed_zero_infinities_and_nan(self):
        neighbors = [0., -0., math.nextafter(.5, 0.), .5, math.nextafter(.5, 1.),
                     float("inf"), float("-inf"), float("nan")]
        for value in neighbors:
            self.compare(self.node([(.5, 10, 0.), (value, 10, 0.)]))
            self.compare(self.node([(.5, 10, value), (.5, 10, 0.)]))
        self.assertEqual(self.compare(self.node([(float("nan"), 1, 0.)])), (None, None))
        self.assertEqual(self.compare(self.node([(0., 1, 1e9)]), 0.), (None, None))

    def test_visits_arbitrary_integers_overflow_and_invalid_counts(self):
        for visits in (2**53-1, 2**53+1, 2**80, 2**1100, -1, -2):
            self.compare(self.node([(.5, visits, .125), (.5, 0, 0.)]))
        self.compare(self.node([(.5, 0, 2**1100)]))
        self.compare(self.node([(.5, 1, 2**1100)]))

    def test_numpy_scalar_arithmetic_remains_dynamic(self):
        for dtype in (np.float32, np.float64):
            self.compare(self.node([(dtype(.5), 11, dtype(.2)), (dtype(.5), 10, dtype(.1))]), dtype(1.5))
        self.compare(self.node([(.5, np.int64(4), -.2), (.5, 7, .3)]))

    def test_random_statistics_and_adjacent_float_priors(self):
        rng = random.Random(20261009)
        for _ in range(1000):
            size = rng.randrange(1, 65)
            stats = [(rng.random(), rng.randrange(3201), rng.uniform(-100., 100.)) for _ in range(size)]
            self.compare(self.node(stats), rng.uniform(0., 5.))

    def test_live_q_value_and_math_sqrt_changes(self):
        root = self.node([(.2, 10, 4.), (.8, 9, -.7)])
        with mock.patch.object(az.PUCTNode, "q_value", lambda child: child.prior):
            self.compare(root)
        with mock.patch.object(az.math, "sqrt", lambda total: .125):
            self.compare(root)

    def test_global_sum_override_falls_back(self):
        root = self.node([(.2, 10, 4.), (.8, 9, -.7)])
        with mock.patch.object(az, "sum", lambda values: 999, create=True):
            self.assertFalse(self.selector.supports(root))
            self.compare(root)

    def test_subclass_and_custom_mapping_fall_back(self):
        class SubNode(az.PUCTNode):
            pass

        class SubDict(dict):
            pass

        root = SubNode(1., 0)
        root.children = self.node([(.5, 1, .3)]).children
        self.assertFalse(self.selector.supports(root))
        # Local subclass cannot be pickled; check the fallback winner directly.
        a, b = experiment.ORIGINAL(root, 1.5), self.selector.select_child(root, 1.5)
        self.assertEqual(a[0], b[0])
        self.assertIs(a[1], b[1])
        root = self.node([(.5, 1, .3)])
        root.children = SubDict(root.children)
        self.assertFalse(self.selector.supports(root))
        a, b = experiment.ORIGINAL(root, 1.5), self.selector.select_child(root, 1.5)
        self.assertEqual(a[0], b[0])
        self.assertIs(a[1], b[1])

    def test_patch_binds_as_a_method_and_restores(self):
        root = self.node([(.2, 10, 4.), (.8, 9, -.7)])
        expected = experiment.ORIGINAL(root, 1.5)
        self.assertTrue(self.selector.supports(root))
        with mock.patch.object(az.PUCTNode, "select_child", self.selector.select_child):
            self.assertEqual(root.select_child(1.5), expected)
        self.assertIs(az.PUCTNode.select_child, experiment.ORIGINAL)


if __name__ == "__main__":
    unittest.main()
