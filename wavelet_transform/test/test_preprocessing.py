import unittest
import numpy as np

from wavelet_transform.utils.preprocessing import preprocess, add_noise


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_normalizes(self):
        # simple array with known mean and std
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        out = preprocess(data, window=False)

        # mean should be ~0 and std ~1
        self.assertAlmostEqual(float(np.mean(out)), 0.0, places=12)
        self.assertAlmostEqual(float(np.std(out)), 1.0, places=12)

    def test_preprocess_window_changes_values_and_centres(self):
        # check that window=True preserves length and centres the signal
        data = np.linspace(0.0, 1.0, 101)

        out_no_win = preprocess(data, window=False)
        out_win = preprocess(data, window=True)

        # length preserved
        self.assertEqual(len(out_win), len(data))

        # still zero mean after windowing (function recentres after applying window)
        self.assertAlmostEqual(float(np.mean(out_win)), 0.0, places=12)

        # windowed output should not be identical to the plain normalised output
        self.assertFalse(np.allclose(out_win, out_no_win))

    def test_add_noise_is_reproducible_with_seed(self):
        # Use a fixed RNG seed so that the produced noise is deterministic
        data = np.linspace(0.0, 1.0, 50)
        snr = 10.0

        np.random.seed(42)
        noisy = add_noise(data, snr)

        # Re-seed and manually generate the expected noise using the same formula
        np.random.seed(42)
        data_to_noise_power_ratio = 10 ** (snr / 10)
        power_noise = np.var(data) / data_to_noise_power_ratio
        std_noise = np.sqrt(power_noise)
        expected_noise = np.random.normal(0, std_noise, len(data))
        expected = data + expected_noise

        # exact equality should hold if RNG seed is the same
        self.assertTrue(np.allclose(noisy, expected))


if __name__ == "__main__":
    unittest.main()
