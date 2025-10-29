import unittest
import numpy as np
import pandas as pd

from wavelet_transform.utils.wavelet_features import extract_wavelet_features


class TestWaveletFeatures(unittest.TestCase):
    def test_extract_wavelet_features_basic(self):
        # simple sinusoid
        t = np.linspace(0, 1, 256, endpoint=False)
        x = np.sin(2 * np.pi * 5 * t)

        df = extract_wavelet_features(x, levels=3, wavelet="haar", mode="symmetric")
        self.assertIsInstance(df, pd.DataFrame)
        # expect at least one row and some columns
        self.assertGreater(df.shape[0], 0)
        self.assertGreater(df.shape[1], 0)
        # no NaNs in numerical columns
        num = df.select_dtypes(include=[np.number]).to_numpy()
        self.assertFalse(np.isnan(num).any())

    def test_keep_ratio_trimming(self):
        # Build a longer signal so trimming for 'same'/'valid' is non-zero
        t = np.linspace(0, 1, 1024, endpoint=False)
        x = np.sin(2 * np.pi * 50 * t)

        # Default transform object is created inside extract_wavelet_features when include_tr_obj=True
        df, tr = extract_wavelet_features(x, levels=4, wavelet="db4", mode="symmetric", include_tr_obj=True)

        # pick a non-base node to ensure level-dependent trimming is exercised
        node = tr.get_level(3, order="natural")[0]

        # full: no trimming
        tr.keep_ratio = "full"
        expected_full = len(node.data)
        self.assertEqual(len(node.data_keep), expected_full)

        # same: use node.get_trim_len to compute lim, then expected length
        tr.keep_ratio = "same"
        lim_same = node.get_trim_len(node.get_filter_len(), node.level, "same")
        if lim_same == 0:
            expected_same = len(node.data)
        else:
            expected_same = len(node.data) - 2 * lim_same
        self.assertEqual(len(node.data_keep), expected_same)

        # valid: compute lim and expected length
        tr.keep_ratio = "valid"
        lim_valid = node.get_trim_len(node.get_filter_len(), node.level, "valid")
        if lim_valid == 0:
            expected_valid = len(node.data)
        else:
            expected_valid = len(node.data) - 2 * lim_valid
        self.assertEqual(len(node.data_keep), expected_valid)

        # numeric ratio: compute lim using code's formula and compare
        ratio = 0.5
        tr.keep_ratio = ratio
        lim_num = int((1 - ratio) * node.data_len / 2)
        if lim_num == 0:
            expected_num = len(node.data)
        else:
            expected_num = len(node.data) - 2 * lim_num
        self.assertEqual(len(node.data_keep), expected_num)

    def test_valid_convolution_equivalence(self):
        """Reconstruct node coefficients by performing valid convolutions and
        downsampling along the node path and compare to node.data_keep when
        keep_ratio='valid'. This checks that trimming reproduces a true 'valid'
        convolution result.
        """
        t = np.linspace(0, 1, 1024, endpoint=False)
        x = np.sin(2 * np.pi * 50 * t)

        df, tr = extract_wavelet_features(x, levels=4, wavelet="db4", mode="symmetric", include_tr_obj=True)

        # test a few leaf nodes
        leaf_nodes = tr.get_level(3, order="natural")
        for node in leaf_nodes[:3]:
            # build the original parent data from the transform's base node
            parent_data = tr.base_node.data.copy()

            # iterate through the path and apply valid convolution + downsample
            path = node.path
            filter_bank = tr.filter_bank
            coeffs = parent_data
            for char in path:
                filt = filter_bank[0] if char == "a" else filter_bank[1]
                # replicate NodeBase.filter2 behavior exactly
                length = len(filt) - 1
                import pywt

                data_pad = pywt.pad(coeffs, (length, length), tr.mode)
                data_conv = np.convolve(data_pad, filt, mode="valid")
                lim = 2 * (len(data_conv) // 2)
                coeffs = data_conv[1:lim:2]

            # Now coeffs is the simulated 'full-convolution-downsampled' result for that node
            # It should equal node.data
            self.assertTrue(np.allclose(coeffs, node.data, atol=1e-12, rtol=1e-9))

            # Compare trimmed version (valid) to node.data_keep when keep_ratio='valid'
            tr.keep_ratio = "valid"  # type: ignore
            kept = node.data_keep
            lim_trim = node.get_trim_len(node.get_filter_len(), node.level, "valid")
            if lim_trim == 0:
                expected_kept = coeffs
            else:
                expected_kept = coeffs[lim_trim:-lim_trim]

            self.assertEqual(len(kept), len(expected_kept))
            if len(kept) > 0:
                self.assertTrue(np.allclose(kept, expected_kept, atol=1e-8, rtol=1e-5))

    def test_same_convolution_equivalence(self):
        """Check that setting keep_ratio='same' returns the centered slice of
        the node coefficients determined by get_trim_len(..., 'same')."""
        t = np.linspace(0, 1, 1024, endpoint=False)
        x = np.sin(2 * np.pi * 50 * t)

        df, tr = extract_wavelet_features(x, levels=4, wavelet="db4", mode="symmetric", include_tr_obj=True)

        leaf_nodes = tr.get_level(3, order="natural")
        for node in leaf_nodes[:3]:
            # node.data is the full downsampled coefficients (as computed by filter2)
            coeffs = node.data

            # compute expected kept slice using 'same' trimming
            lim_same = node.get_trim_len(node.get_filter_len(), node.level, "same")
            if lim_same == 0:
                expected_kept = coeffs
            else:
                expected_kept = coeffs[lim_same:-lim_same]

            tr.keep_ratio = "same"
            kept = node.data_keep

            self.assertEqual(len(kept), len(expected_kept))
            if len(kept) > 0:
                self.assertTrue(np.allclose(kept, expected_kept, atol=1e-8, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
