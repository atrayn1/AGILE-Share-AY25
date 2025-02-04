import pytest
import torch
from agile.graphing import mergeResults
import pandas as pd
# Test matrices
testMatrix1 = pd.DataFrame([
    [5, 10, 15],
    [15, 10, 5],
    [5, 5, 5]
])

testMatrix2 = pd.DataFrame([
    [10, 4, 1],
    [2, 6, 8],
    [9, 3, 5]
])

def test_mergeResults_x_0():
    expected = torch.tensor([[10.,  4.,  1.],
                             [ 2.,  6.,  8.],
                             [ 9.,  3.,  5.]])
    result = mergeResults(testMatrix1, testMatrix2, 1)
    assert torch.allclose(result, expected, atol=1e-4), "Failed for x = 0"

def test_mergeResults_x_1():
    expected = torch.tensor([[ 5., 10., 15.],
                             [15., 10.,  5.],
                             [ 5.,  5.,  5.]])
    result = mergeResults(testMatrix1, testMatrix2, 0)
    assert torch.allclose(result, expected, atol=1e-4), "Failed for x = 1"

def test_mergeResults_x_05():
    expected = torch.tensor([[7.5000, 7.0000, 8.0000],
                             [8.5000, 8.0000, 6.5000],
                             [7.0000, 4.0000, 5.0000]])
    result = mergeResults(testMatrix1, testMatrix2, 0.5)
    assert torch.allclose(result, expected, atol=1e-4), "Failed for x = 0.5"
