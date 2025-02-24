import pytest
import numpy as np
from agile.graphing import mergeResults

# Test matrices as NumPy arrays
testMatrix1 = np.array([
    [5, 10, 15],
    [15, 10, 5],
    [5, 5, 5]
])

testMatrix2 = np.array([
    [10, 4, 1],
    [2, 6, 8],
    [9, 3, 5]
])

def test_mergeResults_x_0():
    expected = np.array([
        [2.0000, 0.4000, 0.0667],
        [0.1333, 0.6000, 1.6000],
        [1.8000, 0.6000, 1.0000]
    ])
    result = mergeResults(testMatrix1, testMatrix2, 0)
    print(result)
    assert np.allclose(result, expected, atol=1e-4), "Failed for x = 0"
