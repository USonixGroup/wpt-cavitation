import unittest
import numpy as np
import pywt
from wavelet_transform.wpt.transform import dwpt
from wavelet_transform.utils.synthetic_data import sound_wave
from wavelet_transform.utils.statistical_features import energy
import wavelet_transform.wpt.inverse_transform as inv
from wavelet_transform.utils.extensions import fft

class TestWaveletTransforms(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Setup common test parameters
        self.levels = 6
        self.wavelet = 'db10'
        self.sf = 50e6  # sampling frequency
        self.freq = 2e6  # signal frequency
        self.amp_sin_waves = 1
        self.phase = 0
        self.signal_duration = 1000/self.sf
        self.mode = "symmetric"

        # Generate test signal
        self.data, self.t_plot = sound_wave(
            self.amp_sin_waves, self.freq, self.phase, 
            self.signal_duration, sampling_rate=self.sf
        )
        
        # Create transform object
        self.transform_object = dwpt(self.data, self.levels, self.wavelet, self.mode)
        
    def test_pywt_comparison(self):
        """Test that our transform matches PyWavelets output at multiple levels"""
        wav = pywt.Wavelet(self.wavelet)  # type: ignore
        
        # Test first three levels
        for level in range(1, 4):
            # Get our transform results for this level
            our_nodes = self.transform_object.get_level(level)
            
            # For each node at this level
            data = self.data.copy()
            pywp = pywt.WaveletPacket(data, wav, maxlevel=self.levels, mode=self.mode)
            
            # Compare results
            for node in our_nodes:
                pywt_result = pywp[node.path].data
                np.testing.assert_array_almost_equal(
                    node.data, pywt_result,
                    err_msg=f"Mismatch at level {level}, node path '{node.path}'"
                )

    def test_energy_conservation_zero_padding(self):
        """Test energy conservation with zero padding"""
        # Create transform with zero padding
        transform_zero = dwpt(self.data, self.levels, self.wavelet, mode='zero')
        
        # Calculate total energy of input signal
        signal_energy = energy(self.data)
        
        # Calculate total energy of leaf nodes
        leaf_nodes = transform_zero.get_leaf_nodes()
        leaf_energy = sum(node.energy for node in leaf_nodes)
        
        # Compare energies
        self.assertAlmostEqual(signal_energy, leaf_energy, places=10,
            msg="Energy not conserved in leaf nodes with zero padding")
        
        # Test energy conservation at each level
        for level in range(1, self.levels + 1):
            level_nodes = transform_zero.get_level(level)
            level_energy = sum(node.energy for node in level_nodes)
            self.assertAlmostEqual(signal_energy, level_energy, places=10,
                msg=f"Energy not conserved at level {level}")

    def test_energy_conservation_dwt(self):
        """Test energy conservation in discrete wavelet transform"""

        # Generate results with zero padding
        transform_zero = dwpt(self.data, self.levels, self.wavelet, mode='zero')

        # Get DWT results
        dwt_nodes = transform_zero.dwt_results()
        dwt_energy = sum(node.energy for node in dwt_nodes)
        
        # Compare with original signal energy
        signal_energy = energy(self.data)
        self.assertAlmostEqual(signal_energy, dwt_energy, places=10,
            msg="Energy not conserved in DWT decomposition")

    def test_inverse_transform_accuracy(self):
        """Test accuracy of inverse transform"""
        # Get inverse transform
        inverse_data = inv.inv_dwpt(self.transform_object)
        
        # Compare with original signal
        np.testing.assert_array_almost_equal(self.data, inverse_data,
            err_msg="Inverse transform does not recover original signal")
        
        # Test energy conservation
        orig_energy = energy(self.data)
        inv_energy = energy(inverse_data)
        self.assertAlmostEqual(orig_energy, inv_energy, places=10,
            msg="Energy not conserved in inverse transform")

    def test_dwt_inverse(self):
        """Test if discrete wavelet transform inverse recovers original signal"""
        dwt_inverse = inv.inv_dwt(self.transform_object)
        error = sum(self.data - dwt_inverse)
        self.assertAlmostEqual(error, 0, places=10, 
            msg="DWT inverse transform error too large")

    def test_dwpt_inverse(self):
        """Test if discrete wavelet packet transform inverse recovers original signal"""
        dwpt_inverse = inv.inv_dwpt(self.transform_object)
        error = sum(np.array(self.data) - dwpt_inverse)
        self.assertAlmostEqual(error, 0, places=10,
            msg="DWPT inverse transform error too large")

    def test_frequency_calculation(self):
        """Test frequency calculation in transform object using uniform frequency spectrum"""
        # Create a signal with uniform frequency spectrum like in the notebook
        duration = 30
        sampling_rate = 10
        n_samples = int(duration * sampling_rate)
        
        # Create uniform frequency spectrum
        freq_spectrum = np.ones(n_samples, dtype=complex)
        signal = np.fft.ifft(freq_spectrum)
        signal = np.abs(signal)
        
        # Create transform object with this signal
        levels = 4
        transform = dwpt(signal, levels, "db1", mode="zero")
        nodes = transform.get_level(order="freq")

        freq_peaks = []
        
        # Calculate frequencies for each node
        for node in nodes:
            # Reconstruct signal from this node
            rec = inv.inv_dwpt(transform, nodes=node)
            fft_results, frequencies = fft(rec, sampling_rate)
            fft_results = abs(fft_results)
            
            # Calculate average frequency from FFT
            norm_fft_results = fft_results / sum(fft_results)
            
            # Calculate center frequency of node
            cent_freq = 1/2 * sum(node.numeric_freq_range) * sampling_rate/2
            
            # Find peak frequency
            freq_peak = frequencies[np.argmax(norm_fft_results)]
            freq_peaks.append(freq_peak)

            # Test that peak frequency is close to center frequency
            # Allow for 0.2 difference in Hz
            self.assertLess(abs(freq_peak - cent_freq), 0.2,
                msg=f"Node {node.path}: Peak frequency {freq_peak:.2f} Hz too far from center frequency {cent_freq:.2f} Hz")
            
        # Verify frequencies are monotonically increasing
        self.assertEqual(freq_peaks, sorted(freq_peaks),
            msg="Center frequencies should be in ascending order")

if __name__ == '__main__':
    unittest.main()