import pytest
import torch
from agile.graphing import mergeResults
import pandas as pd
# Test matrices
testMatrix1 = [
    [5, 10, 15],
    [15, 10, 5],
    [5, 5, 5]
]

testMatrix2 = [
    [10, 4, 1],
    [2, 6, 8],
    [9, 3, 5]
]

def test_mergeResults_x_0():
    expected = torch.tensor([[2.0000, 0.4000, 0.0667],
        [0.1333, 0.6000, 1.6000],
        [1.8000, 0.6000, 1.0000]])
    result = mergeResults(testMatrix1, testMatrix2, 0)
    print(result)
    assert torch.allclose(result, expected, atol=1e-4), "Failed for x = 0"
