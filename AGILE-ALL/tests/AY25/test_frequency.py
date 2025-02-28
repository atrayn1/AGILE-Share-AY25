import pytest
from datetime import datetime

# Import the function to test
from agile.graphing import frequencyOfColocation  # Replace 'your_module' with the actual module name

def test_no_overlap():
    """Test when there is no overlap at all between the two period lists."""
    periods1 = [((datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 10, 0)), (0, 0))]
    periods2 = [((datetime(2024, 1, 1, 12, 20, 0), datetime(2024, 1, 1, 12, 30, 0)), (0, 0))]
    
    assert frequencyOfColocation(periods1, periods2, 60, 0) == 0

def test_full_overlap():
    """Test when one period is completely within another."""
    periods1 = [((datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 30, 0)), (0, 0))]
    periods2 = [((datetime(2024, 1, 1, 12, 5, 0), datetime(2024, 1, 1, 12, 25, 0)), (0, 0))]
    
    assert frequencyOfColocation(periods1, periods2, 60, 0) == 1

def test_partial_overlap():
    """Test when two periods partially overlap."""
    periods1 = [((datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 20, 0)), (0, 0))]
    periods2 = [((datetime(2024, 1, 1, 12, 10, 0), datetime(2024, 1, 1, 12, 30, 0)), (0, 0))]
    
    assert frequencyOfColocation(periods1, periods2, 600, 0) == 1  # 10 min (600 sec) overlap

def test_multiple_overlaps():
    """Test when multiple colocations occur."""
    periods1 = [
        ((datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 20, 0)), (0, 0)),
        ((datetime(2024, 1, 1, 13, 0, 0), datetime(2024, 1, 1, 13, 20, 0)), (0, 0))
    ]
    periods2 = [
        ((datetime(2024, 1, 1, 12, 10, 0), datetime(2024, 1, 1, 12, 30, 0)), (0, 0)),
        ((datetime(2024, 1, 1, 13, 5, 0), datetime(2024, 1, 1, 13, 15, 0)), (0, 0))
    ]
    
    assert frequencyOfColocation(periods1, periods2, 300, 0) == 2  # Both overlaps are at least 5 min (300 sec)

def test_borderline_overlap():
    """Test when overlap is exactly at the x_time threshold."""
    periods1 = [((datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 10, 0)), (0, 0))]
    periods2 = [((datetime(2024, 1, 1, 12, 5, 0), datetime(2024, 1, 1, 12, 15, 0)), (0, 0))]
    
    assert frequencyOfColocation(periods1, periods2, 300, 0) == 1  # 5 min (300 sec) overlap

def test_no_colocation_due_to_threshold():
    """Test when there is overlap but does not meet x_time threshold."""
    periods1 = [((datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 10, 0)), (0, 0))]
    periods2 = [((datetime(2024, 1, 1, 12, 5, 0), datetime(2024, 1, 1, 12, 7, 0)), (0, 0))]
    
    assert frequencyOfColocation(periods1, periods2, 180, 0) == 0  # Only 2 min overlap, below threshold

def test_identical_periods():
    """Test when both periods are exactly the same."""
    periods1 = [((datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 30, 0)), (0, 0))]
    periods2 = [((datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 30, 0)), (0, 0))]
    
    assert frequencyOfColocation(periods1, periods2, 1800, 0) == 1  # Full 30 min overlap

def test_one_empty_list():
    """Test when one list of periods is empty."""
    periods1 = [((datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 30, 0)), (0, 0))]
    periods2 = []
    
    assert frequencyOfColocation(periods1, periods2, 60, 0) == 0

def test_both_empty_lists():
    """Test when both lists are empty."""
    periods1 = []
    periods2 = []
    
    assert frequencyOfColocation(periods1, periods2, 60, 0) == 0

def test_large_time_gap():
    """Test when periods have a large time gap between them."""
    periods1 = [((datetime(2024, 1, 1, 10, 0, 0), datetime(2024, 1, 1, 10, 30, 0)), (0, 0))]
    periods2 = [((datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 30, 0)), (0, 0))]
    
    assert frequencyOfColocation(periods1, periods2, 60, 0) == 0

def test_multiple_periods_with_no_overlap():
    """Test when multiple periods exist but none overlap."""
    periods1 = [
        ((datetime(2024, 1, 1, 10, 0, 0), datetime(2024, 1, 1, 10, 10, 0)), (0, 0)),
        ((datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 10, 0)), (0, 0))
    ]
    periods2 = [
        ((datetime(2024, 1, 1, 11, 0, 0), datetime(2024, 1, 1, 11, 10, 0)), (0, 0)),
        ((datetime(2024, 1, 1, 13, 0, 0), datetime(2024, 1, 1, 13, 10, 0)), (0, 0))
    ]
    
    assert frequencyOfColocation(periods1, periods2, 60, 0) == 0