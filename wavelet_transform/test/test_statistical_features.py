import unittest
import numpy as np
import scipy.stats
from wavelet_transform.utils.statistical_features import (
    crest_factor, energy, log_energy, avg_energy, log_avg_energy,
    kurtosis, skewness, abs_skewness, entropy, spectral_entropy
)

class TestStatisticalFeatures(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create various test signals
        np.random.seed(42)
        t = np.linspace(0, 1, 1000)
        
        # Simple sine wave
        self.sine = np.sin(2 * np.pi * 10 * t)
        
        # Gaussian noise
        self.noise = np.random.normal(0, 1, 1000)
        
        # Mixed signal
        self.mixed = self.sine + 0.1 * self.noise
        
        # Known distributions for testing
        self.uniform = np.random.uniform(0, 1, 1000)
        self.gaussian = np.random.normal(0, 1, 1000)
        
        # Signal with known energy
        self.unit_energy = np.array([1.0, 0.0, 0.0, 0.0])  # Energy = 1

    def test_energy_metrics(self):
        """Test energy-related metrics"""
        # Test energy calculation
        self.assertAlmostEqual(energy(self.unit_energy), 1.0)
        self.assertGreater(energy(self.sine), 0)
        
        # Test average energy
        self.assertAlmostEqual(avg_energy(self.unit_energy), 0.25)  # 1.0/4
        
        # Test log energy
        expected_log = 10 * np.log10(1.0)
        self.assertAlmostEqual(log_energy(self.unit_energy), expected_log)
        
        # Test log average energy
        expected_log_avg = 10 * np.log10(0.25)
        self.assertAlmostEqual(log_avg_energy(self.unit_energy), expected_log_avg)

    def test_distribution_metrics(self):
        """Test distribution-related metrics"""
        # Test kurtosis against scipy implementation
        our_kurt = kurtosis(self.gaussian)
        scipy_kurt = scipy.stats.kurtosis(self.gaussian, fisher=False)
        self.assertAlmostEqual(our_kurt, scipy_kurt)
        
        # Test skewness against scipy implementation
        our_skew = skewness(self.gaussian)
        scipy_skew = scipy.stats.skew(self.gaussian)
        self.assertAlmostEqual(our_skew, scipy_skew)
        
        # Test absolute skewness
        self.assertAlmostEqual(abs_skewness(self.gaussian), abs(scipy_skew))

    def test_distribution_metrics_edge_cases(self):
        """Test edge cases for distribution metrics"""
        # Test with constant array
        constant = np.ones(100)
        self.assertTrue(np.isnan(kurtosis(constant)))  # Definition of kurtosis for constant signal
        self.assertTrue(np.isnan(skewness(constant)))  # Constant signal has no skewness
        
        # Test with single value array
        single = np.array([1.0])
        self.assertTrue(np.isnan(kurtosis(single)))
        self.assertTrue(np.isnan(skewness(single)))
            
        # Test with empty array
        empty = np.array([])
        self.assertTrue(np.isnan(kurtosis(empty)))
        self.assertTrue(np.isnan(skewness(empty)))
            
        # Test with array containing NaN
        nan_array = np.array([1.0, np.nan, 3.0])
        self.assertTrue(np.isnan(kurtosis(nan_array)))
        self.assertTrue(np.isnan(skewness(nan_array)))
        
        # Test with array containing Inf
        inf_array = np.array([1.0, np.inf, 3.0])
        self.assertTrue(np.isnan(kurtosis(inf_array)))
        self.assertTrue(np.isnan(skewness(inf_array)))

    def test_crest_factor(self):
        """Test crest factor calculation"""
        # For sine wave, crest factor should be sqrt(2)
        expected_cf = np.sqrt(2)
        calculated_cf = crest_factor(self.sine)
        self.assertAlmostEqual(calculated_cf, expected_cf, places=2)
        
        # Crest factor should be >= 1 for any signal
        self.assertGreaterEqual(crest_factor(self.noise), 1.0)
        self.assertGreaterEqual(crest_factor(self.mixed), 1.0)

    def test_entropy(self):
        """Test entropy calculations"""
        # Test entropy of uniform distribution
        uniform_energy = np.ones(100) / 10  # Normalized uniform energy
        uniform_entropy = entropy(uniform_energy)
        expected_entropy = np.log(100)  # Maximum entropy for 100 elements
        self.assertAlmostEqual(uniform_entropy, expected_entropy, places=2)
        
        # Test entropy of concentrated energy
        concentrated = np.zeros(100)
        concentrated[0] = 1  # All energy in one coefficient
        self.assertAlmostEqual(entropy(concentrated), 0)
        
        # Test with pre-computed total energy
        total_energy = 2.0
        scaled_array = np.array([1.0, 1.0])
        self.assertAlmostEqual(
            entropy(scaled_array, total_energy),
            -2 * (0.5 * np.log(0.5))  # Expected entropy for equal split
        )

    def test_spectral_entropy(self):
        """Test spectral entropy calculation"""
        # Pure sine should have low spectral entropy
        sine_entropy = spectral_entropy(self.sine)
        
        # White noise should have high spectral entropy
        noise_entropy = spectral_entropy(self.noise)
        
        # Noise should have higher entropy than sine
        self.assertGreater(noise_entropy, sine_entropy)
        
        # Both should be between 0 and log(N)
        max_entropy = np.log(len(self.sine))
        self.assertGreater(sine_entropy, 0)
        self.assertLess(sine_entropy, max_entropy)
        self.assertGreater(noise_entropy, 0)
        self.assertLess(noise_entropy, max_entropy)

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test zero array
        zero_array = np.zeros(10)
        self.assertEqual(energy(zero_array), 0)
        self.assertEqual(entropy(zero_array), 0)
        
        # Test single value
        single = np.array([1.0])
        self.assertEqual(energy(single), 1.0)
        self.assertEqual(entropy(single), 0)
        
        # Test invalid inputs
        with self.assertRaises(ValueError):
            entropy([1, 2, 3])  # Not numpy array
        
        with self.assertRaises((ValueError, TypeError)):
            entropy(np.array([]))  # Empty array

if __name__ == '__main__':
    unittest.main()