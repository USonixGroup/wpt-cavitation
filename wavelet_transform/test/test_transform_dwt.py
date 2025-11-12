import unittest
import numpy as np
import pywt

from wavelet_transform.wpt.transform import dwt, calc_max_level


class TestDWT(unittest.TestCase):
    def test_dwt_basic_levels(self):
        # Use a power-of-two length so calc_max_level is an integer
        data = np.arange(8)
        max_lvl = 3
        tr = dwt(data, max_lvl, wavelet="db1", mode="symmetric")

        # transform should have levels 0..max_lvl
        self.assertEqual(tr.max_level(), max_lvl)

        # level 0 should have one node, subsequent levels should each have 2 nodes
        for lvl in range(max_lvl + 1):
            nodes = tr.get_level(lvl)
            if lvl == 0:
                self.assertEqual(len(nodes), 1)
            else:
                self.assertEqual(len(nodes), 2)

    def test_dwt_matches_pywt_coefficients(self):
        # Check that coefficients produced by our dwt match pywt.wavedec for the same wavelet/mode
        data = np.linspace(0, 1, 16)
        max_lvl = 3
        wavelet = "db1"
        mode = "symmetric"

        tr = dwt(data, max_lvl, wavelet=wavelet, mode=mode)

        # pywt returns [cA_n, cD_n, cD_{n-1}, ..., cD1]
        coeffs = pywt.wavedec(data, wavelet, mode=mode, level=max_lvl)

        # Compare the final approximation (cA_n) with the node at path 'a'*max_lvl
        approx_node = tr.get_node(max_lvl, "a" * max_lvl)
        np.testing.assert_allclose(approx_node.data, coeffs[0], rtol=1e-6, atol=1e-8)

        # Compare detail coefficients cD_k with corresponding nodes
        # coeffs[1] is cD_max, coeffs[-1] is cD1
        for k in range(1, max_lvl + 1):
            # node path for detail at level k is 'a'*(k-1) + 'd'
            node = tr.get_node(k, "a" * (k - 1) + "d")
            # index in pywt list for cD_k
            idx = 1 + (max_lvl - k)
            np.testing.assert_allclose(node.data, coeffs[idx], rtol=1e-6, atol=1e-8)

    def test_dwt_default_max_level_and_list_input(self):
        # Passing a python list should be accepted and treated like an ndarray
        data_list = list(range(8))
        tr = dwt(data_list, None, wavelet="db1", mode="symmetric")

        # default max level should match calc_max_level
        expected = calc_max_level(np.array(data_list))
        self.assertEqual(tr.max_level(), expected)

        # spot-check node counts
        self.assertEqual(len(tr.get_level(0)), 1)
        if expected >= 1:
            self.assertEqual(len(tr.get_level(1)), 2)

    def test_dwt_too_large_level_raises(self):
        data = np.arange(8)
        # calc_max_level for len 8 is 3, so 4 should raise
        with self.assertRaises(ValueError):
            dwt(data, 4)


if __name__ == "__main__":
    unittest.main()
