import unittest
import numpy as np
import wavelet_transform.wpt.transform as wpt

class TestShannonEntropyDecomposition(unittest.TestCase):
    def test_ones(self):
        data = np.ones(16)
        tr = wpt.dwpt(data, None, 'haar', "zero", min_entropy=True)
        leaf_nodes = tr.get_leaf_nodes()
        leaf_node_value = [(node.level, node.node_number) for node in leaf_nodes]
        expected_leaf_nodes = [(1, 1), (2, 1),(3, 1), (4, 0), (4, 1)]
        self.assertEqual(leaf_node_value, expected_leaf_nodes)

    def test_sine_wave(self):
        # Generate high frequency sine wave
        fs = 1e9  # sampling frequency
        f = 1.1e6  # sine wave frequency
        t = np.linspace(0, 1e-3, int(fs * 1e-3), endpoint=False)
        data = np.sin(2 * np.pi * f * t)

        tr = wpt.dwpt(data, 6, 'haar', "zero", min_entropy=True)
        tr.plot_tree()
        leaf_nodes = tr.get_leaf_nodes()
        leaf_node_value = [(node.level, node.node_number) for node in leaf_nodes]
        expected_leaf_nodes = [(3, 7), (4, 11), (4, 13), (5, 15), (5, 25), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (6, 24), (6, 25), (6, 26), (6, 27), (6, 28), (6, 29), (6, 32), (6, 33), (6, 34), (6, 35), (6, 36), (6, 37), (6, 38), (6, 39), (6, 40), (6, 41), (6, 42), (6, 43), (6, 48), (6, 49)]
        self.assertEqual(leaf_node_value, expected_leaf_nodes)


if __name__ == '__main__':
    unittest.main()