import unittest
import numpy as np
import pywt

from wavelet_transform.wpt.transform import dwt
import wavelet_transform.wpt.inverse_transform as inv_mod


class TestInverseNode(unittest.TestCase):
    def test_inv_node_matches_pywt_upcoef(self):
        data = np.linspace(0, 1, 32)
        max_lvl = 4
        wavelet = "db1"
        mode = "symmetric"

        tr = dwt(data, max_lvl, wavelet=wavelet, mode=mode)

        # test a handful of nodes including deepest approx and a detail
        nodes_to_test = [
            tr.get_node(max_lvl, "a" * max_lvl),
            tr.get_node(max_lvl, "a" * (max_lvl - 1) + "d"),
        ]

        for node in nodes_to_test:
            our_rec = inv_mod.inv_node(node)

            # iterative pywt.upcoef reconstruction: start from node coefficients
            py_rec = node.data.copy()
            for ad in reversed(node.path):
                part = "a" if ad == "a" else "d"
                # upcoef reconstructs one level when level=1
                py_rec = pywt.upcoef(part, py_rec, wavelet, level=1)

            # Trim or pad py_rec to same length as our_rec and compare
            L = len(our_rec)
            py_rec_cut = py_rec[:L]

            np.testing.assert_allclose(our_rec, py_rec_cut, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
