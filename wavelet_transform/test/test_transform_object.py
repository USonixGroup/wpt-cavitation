import unittest
import numpy as np
from wavelet_transform.wpt.transform_object import Transform
from wavelet_transform.wpt.transform_node import Node, BaseNode

class TestWaveletTransform(unittest.TestCase):
    def setUp(self):
        """Set up test data and common transform parameters"""
        # Create a simple test signal: sum of two sine waves
        t = np.linspace(0, 1, 1000)
        f1, f2 = 10, 50  # Hz
        self.test_signal = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)
        
        # Common transform parameters
        self.wavelet = 'db4'
        self.mode = 'symmetric'
        
        # Create transform object
        self.transform = Transform(self.test_signal, self.wavelet, self.mode)

    def test_transform_initialization(self):
        """Test basic transform object initialization"""
        # Check basic attributes
        self.assertEqual(self.transform.wavelet, self.wavelet)
        self.assertEqual(self.transform.mode, self.mode)
        np.testing.assert_array_equal(self.transform.data, self.test_signal)
        
        # Check filter bank
        self.assertIsNotNone(self.transform.filter_bank)
        self.assertEqual(len(self.transform.filter_bank), 4)  # lo_dec, hi_dec, lo_rec, hi_rec
        
        # Check base node
        self.assertIsInstance(self.transform.base_node, BaseNode)
        self.assertEqual(len(self.transform.levels), 1)
        self.assertIn(0, self.transform.levels)

    def test_base_node_properties(self):
        """Test base node properties"""
        base_node = self.transform.base_node
        
        # Check attributes
        self.assertEqual(base_node.level, 0)
        self.assertEqual(base_node.path, '')
        self.assertIsNone(base_node.parent)
        
        # Check data
        np.testing.assert_array_equal(base_node.data, self.test_signal)

    def test_node_filtering(self):
        """Test node filtering operations"""
        # Get first level nodes
        base_node = self.transform.base_node
        low_pass = base_node.filter(
            base_node.data.copy(),
            self.transform.filter_bank[0],
            self.transform.mode
        )
        high_pass = base_node.filter(
            base_node.data.copy(),
            self.transform.filter_bank[1],
            self.transform.mode
        )
        
        # Check filtered data
        self.assertEqual(len(low_pass), len(high_pass))

    def test_node_properties(self):
        """Test node statistical properties"""
        # Create a node
        node = Node(self.transform.base_node, 'a', self.transform)
        
        # Test basic properties
        self.assertIsNotNone(node.data)
        self.assertEqual(node.level, 1)
        self.assertEqual(node.path, 'a')
        
        # Test data retention
        self.transform.keep_ratio = 0.8
        retained_data = node.data_keep
        self.assertLess(len(retained_data), len(node.data))

    def test_frequency_ranges(self):
        """Test frequency range calculation for nodes"""
        # Set sampling rate
        fs = 1000  # Hz
        self.transform.sampling_rate = fs
        
        # Create nodes
        approx_node = Node(self.transform.base_node, 'a', self.transform)
        detail_node = Node(self.transform.base_node, 'd', self.transform)
        
        # Check normalized frequency ranges
        self.assertLessEqual(approx_node.norm_frequency_range[1], 0.5)
        self.assertGreaterEqual(detail_node.norm_frequency_range[0], 0.5)
        
        # Check absolute frequency ranges after setting sampling rate
        nyquist = fs/2
        approx_freq = approx_node.norm_frequency_range[1] * nyquist
        detail_freq = detail_node.norm_frequency_range[0] * nyquist
        
        self.assertLessEqual(approx_freq, nyquist/2)
        self.assertGreaterEqual(detail_freq, nyquist/2)

if __name__ == '__main__':
    unittest.main()