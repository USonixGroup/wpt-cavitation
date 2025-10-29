# Wavelet Transform Tests

This directory contains unit tests for the wavelet transform package. The tests cover core transform functionality, statistical features, entropy calculations, and machine learning components.

## Running Tests

You can run the tests in several ways:

1. Run all tests from project root:
```bash
python -m unittest discover wavelet_transform/test
```

2. Run a specific test file:
```bash
python -m unittest wavelet_transform/test/test_dwpt.py
```

3. Run a specific test class:
```bash
python -m unittest wavelet_transform.test.test_dwpt.TestDWPT
```

4. Run a specific test method:
```bash
python -m unittest wavelet_transform.test.test_dwpt.TestDWPT.test_dwpt_inverse
```

## Test Structure

- `test_transform_object.py`: Tests for the base transform object and its properties
- `test_dwpt.py`: Tests for Discrete Wavelet Packet Transform (DWPT) operations
- `test_statistical_features.py`: Tests for statistical feature calculations including:
  - Energy metrics (energy, log energy, average energy)
  - Distribution metrics (kurtosis, skewness)
  - Signal characteristics (crest factor)
  - Edge cases and validation against scipy implementations
- `test_shannon_entropy.py`: Tests for entropy calculations including:
  - Wavelet energy entropy
  - Spectral entropy
  - Validation against theoretical values
- `test_machine_learning.py`: Tests for machine learning integration components
- `conftest.py`: Test configuration and shared fixtures
- `__init__.py`: Makes the test directory a Python package

## Test Coverage

The test suite covers:
1. Core Transform Operations:
   - Basic transform properties
   - Forward and inverse transforms
   - Energy conservation
   - Comparison with PyWavelets implementation

2. Statistical Features:
   - Basic functionality
   - Edge cases (empty arrays, NaN values, etc.)
   - Validation against scipy implementations
   - Integration with wavelet coefficients

3. Entropy Calculations:
   - Correctness of entropy computations
   - Edge cases and boundary conditions
   - Theoretical validations

4. Machine Learning Components:
   - Feature extraction
   - Integration with transform objects

## Adding New Tests

When adding new tests:
1. Create a new test file if testing a new component
2. Follow the naming convention: `test_*.py`
3. Use meaningful test method names that describe what's being tested
4. Add docstrings to test methods explaining:
   - Purpose of the test
   - Input data characteristics
   - Expected outcomes
5. Include both positive and negative test cases
6. Add edge case testing where applicable
7. For numerical computations, validate against known implementations or theoretical values

## Best Practices

1. Keep tests focused and atomic
2. Use descriptive assertion messages
3. Set up test data in `setUp` methods when used by multiple tests
4. Clean up resources in `tearDown` methods if needed
5. Use appropriate tolerances for floating-point comparisons
6. Document any assumptions or special conditions
7. Include performance-critical test cases where relevant