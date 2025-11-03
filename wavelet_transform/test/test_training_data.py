import unittest
import numpy as np
import pandas as pd
from unittest import mock

import wavelet_transform.utils.training_data as td


class TestTrainingData(unittest.TestCase):
    def test_get_training_data_small(self):
        # Set sampling rate and noise std
        sampling_rate = 50e6
        std_noise = 0.01

        # Parameters for the wavelet transform
        wavelet = "db30"
        levels = 6  # max level for the transform
        mode = "symmetric"
        best_basis = None
        min_level = 0

        freq_threshold = 15e6 / (sampling_rate / 2)

        wpt_df = td.get_training_data(
            wavelet,
            sampling_rate,
            levels,
            mode,
            std_noise,
            best_basis=best_basis,
            min_level=min_level,
            freq_threshold=freq_threshold,
        )

        df = wpt_df if isinstance(wpt_df, pd.DataFrame) else wpt_df[0]

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.shape[0], 0)
        num = df.select_dtypes(include=[np.number]).to_numpy()
        self.assertFalse(np.isnan(num).any())


if __name__ == "__main__":
    unittest.main()
