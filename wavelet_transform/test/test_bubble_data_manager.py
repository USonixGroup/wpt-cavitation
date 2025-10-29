import unittest
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import resample as sc_resample

from wavelet_transform.utils.extensions import psd
from wavelet_transform.utils.bubble_data_manager import (
    _get_time_array,
    resample,
    _get_params_from_kwargs,
    generate_data,
    get_data,
)



class TestBubbleDataManager(unittest.TestCase):
    def setUp(self):
        self.test_params = _get_params_from_kwargs(1e-6, 1e5, -100e3, 10)
        self.data = generate_data(500e-6, *self.test_params)  # bd data for testing

        # Resample to 1.1 GHz
        target_fs = 1.1e9
        self.data_resampled = np.asarray(resample(self.data, target_fs))

    def test_time_array_and_resample(self):
        t = _get_time_array(1, 1e5)
        self.assertEqual(t.ndim, 1)
        self.assertGreater(t.size, 0)

        # resample a simple array
        x = np.linspace(0, 1, 1000)
        y = resample(x, 1e6)
        y_arr = np.asarray(y)
        self.assertGreater(y_arr.size, 0)
        self.assertTrue(np.isfinite(y_arr).all())

    def test_generate_data_and_no_nan_short(self):
        # use small n_cycles to keep runtime small
        self.assertIsNotNone(self.data)
        arr = np.asarray(self.data)
        self.assertTrue(np.isfinite(arr).all())
        self.assertFalse(np.isnan(arr).any())

    def test_get_data_returns_array(self):
        # call get_data with small sampling and cycles
        out = get_data(1e-6, 1e5, -1e3, sampling_rate=1e6, n_cycles=1)
        # function may return numpy array or (data,time)
        if isinstance(out, tuple):
            data = out[0]
        else:
            data = out
        self.assertIsNotNone(data)
        arr = np.asarray(data)
        self.assertTrue(np.isfinite(arr).all())

    def test_resample_round_trip_quality(self):
        # create a test tone sampled at 1 MHz and downsample to 250 kHz
        self.assertGreater(self.data_resampled.size, 0)
        y_back = sc_resample(self.data_resampled, len(self.data))

        # metrics
        err = self.data - y_back
        rmse = np.sqrt(np.mean(err ** 2))
        nrmse = rmse / (np.max(np.abs(self.data)) or 1.0)
        corr = np.corrcoef(self.data, y_back)[0, 1]  # type: ignore

        # Expect good reconstruction for band-limited tone
        # tolerances are relaxed because FFT resampling is approximate
        self.assertGreater(corr, 0.98)
        self.assertLess(nrmse, 5e-2)

    def test_energy_preservation_on_resample(self):
        # Calculate average energy of original signal
        orig_energy = np.mean(self.data ** 2)

        # Calculate average energy of resampled signal
        resampled_energy = np.mean(self.data_resampled ** 2)

        # Check that energy is preserved within a tolerance
        self.assertAlmostEqual(orig_energy, resampled_energy, delta=1e-2)

    def test_kurtosis_preservation_on_resample(self):
        # Calculate kurtosis of original signal
        orig_kurtosis = kurtosis(self.data)

        # Calculate kurtosis of resampled signal
        resampled_kurtosis = kurtosis(self.data_resampled)
        # Check that kurtosis is preserved within a tolerance
        self.assertAlmostEqual(orig_kurtosis, resampled_kurtosis, delta=0.1)

    def test_resample_psd_similarity(self):
        fs = int(1e6)
        t = np.arange(0, 0.01, 1 / fs)
        # broadband-ish test signal
        x = np.random.RandomState(0).normal(size=t.shape)

        # Downsample to 250 kHz and resample back
        y = resample(x, 250e3)
        y_back = sc_resample(y, len(x))

        f1, p1 = psd(x, fs)
        f2, p2 = psd(y_back, fs)

        # Calculate cross correlation of the two PSDs
        psd_corr = np.corrcoef(p1[:len(p2)], p2[:len(p1)])[0, 1]
        self.assertGreater(psd_corr, 0.98)

if __name__ == "__main__":
    unittest.main()
