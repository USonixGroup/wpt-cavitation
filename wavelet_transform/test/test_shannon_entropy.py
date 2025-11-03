import unittest
import numpy as np
import wavelet_transform.wpt.transform as wpt

class TestShannonEntropyDecomposition(unittest.TestCase):
    def test_ones(self):
        data = np.ones(16)
        tr = wpt.dwpt(data, None, 'haar', "zero", best_basis="min_entropy")
        leaf_nodes = tr.get_leaf_nodes()
        leaf_node_value = [(node.level, node.node_number) for node in leaf_nodes]
        expected_leaf_nodes = [(1, 1), (2, 1),(3, 1), (4, 0), (4, 1)]
        self.assertEqual(leaf_node_value, expected_leaf_nodes)


if __name__ == '__main__':
    unittest.main()