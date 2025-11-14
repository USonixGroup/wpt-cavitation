import unittest
import numpy as np

from wavelet_transform.wpt.transform import dwpt
from wavelet_transform.wpt.inverse_transform import InverseTransform
from wavelet_transform.wpt import metrics as metrics_mod
import wavelet_transform.wpt.inverse_transform as inv_mod


class TestInfoFunctions(unittest.TestCase):
    def setUp(self):
        # small power-of-two signal for predictable levels
        self.data = np.linspace(0, 1, 32)
        self.max_lvl = 3
        self.wavelet = "db1"
        self.mode = "symmetric"
        self.tr = dwpt(self.data, self.max_lvl, wavelet=self.wavelet, mode=self.mode)

    def test_info_level_int(self):
        df = self.tr.info(level=1, include_node_info=True)
        # Expect rows equal to number of nodes at level 1
        nodes = self.tr.get_level(1)
        self.assertEqual(len(df), len(nodes))
        # include_node_info True should include 'level' and 'node number'
        self.assertIn("level", df.columns)
        self.assertIn("node number", df.columns)

    def test_info_level_leaf(self):
        df_leaf = self.tr.info(level="leaf")
        leaf_nodes = self.tr.get_leaf_nodes()
        self.assertEqual(len(df_leaf), len(leaf_nodes))

    def test_info_level_none_equals_max(self):
        df_none = self.tr.info(level=None)
        df_max = self.tr.info(level=self.tr.max_level())
        self.assertEqual(len(df_none), len(df_max))

    def test_inverse_info_leaf(self):
        inv = InverseTransform(self.tr)
        df_inv = inv.info(level="leaf", include_node_info=True)
        leaf_nodes = self.tr.get_leaf_nodes()
        self.assertEqual(len(df_inv), len(leaf_nodes))
        # inverse info should include metric columns (e.g. 'kurtosis') or at least be non-empty
        self.assertGreaterEqual(len(df_inv.columns), 1)

    def test_invalid_level_raises(self):
        with self.assertRaises(ValueError):
            self.tr.info(level=9999)

    def test_metrics_calculated_correctly_transform_and_inverse(self):
        # Pick some stable metrics to compare
        metric_keys = ["energy", "average_energy", "kurtosis", "spectral_entropy"]

        # Transform.info for level 1
        df = self.tr.info(level=1, metrics=metric_keys)
        nodes = self.tr.get_level(1)
        # Compare first node metrics
        node = nodes[0]
        row = df.iloc[0]

        for m in metric_keys:
            col = m.replace("_", " ")
            expected = metrics_mod.compute_metric_for_data(
                node.data_keep, m, None if m not in metrics_mod.SPECIAL_METRICS 
                else metrics_mod.get_ref(m, self.tr)
            )
            got = row[col]
            if expected is None:
                self.assertIsNone(got)
            else:
                self.assertTrue(np.isfinite(got))
                self.assertAlmostEqual(float(got), float(expected), places=6)

        # InverseTransform.info uses full and trimmed data and references
        inv_tr = InverseTransform(self.tr)
        df_inv = inv_tr.info(level="leaf", metrics=metric_keys)
        leaf_nodes = self.tr.get_leaf_nodes()
        # compare first leaf node
        node = leaf_nodes[0]
        row_inv = df_inv.iloc[0]

        full_inv = inv_mod.inv_node(node)
        # trimmed used by info: default keep_ratio=1. so trimmed==full
        trimmed = inv_tr._trim(full_inv, 1.0)

        for m in metric_keys:
            col = m.replace("_", " ")
            ref = metrics_mod.get_ref(m, self.tr)
            data_for_metric = metrics_mod.get_data(full_inv, trimmed, m)
            expected = metrics_mod.compute_metric_for_data(data_for_metric, m, ref)
            got = row_inv[col]
            if expected is None:
                self.assertIsNone(got)
            else:
                self.assertTrue(np.isfinite(got))
                self.assertAlmostEqual(float(got), float(expected), places=6)


if __name__ == "__main__":
    unittest.main()
