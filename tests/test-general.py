import pytest
import torch
import numpy as np
from ml_training_utils.general import TimeWindowSplitter, DataSplitter

def test_time_window_splitter():
    # Create sample data
    data = torch.randn(2, 3, 12)  # 2 samples, 3 features, 12 time steps
    index_to_info = {0: {'id': 'sample1'}, 1: {'id': 'sample2'}}
    window_size = 4
    
    splitter = TimeWindowSplitter(data, index_to_info, window_size)
    new_data, new_info = splitter.split()
    
    assert new_data.shape == (6, 3, 4)  # 2 samples * 3 windows = 6 total windows
    assert len(new_info) == 6
    
    # Test validation
    is_valid, errors = splitter.validate_split()
    assert is_valid
    assert not errors

def test_data_splitter():
    data = torch.randn(100, 10)  # 100 samples, 10 features
    splitter = DataSplitter(data, val_split=0.2, test_split=0.1)
    
    train_idx, val_idx, test_idx = splitter.split_data()
    
    assert len(train_idx) == 70  # 70% train
    assert len(val_idx) == 20   # 20% validation
    assert len(test_idx) == 10  # 10% test
    
    # Test indices are unique
    all_indices = np.concatenate([train_idx, val_idx, test_idx])
    assert len(np.unique(all_indices)) == len(all_indices)

def test_invalid_window_size():
    data = torch.randn(2, 3, 12)
    index_to_info = {0: {'id': 'sample1'}, 1: {'id': 'sample2'}}
    
    with pytest.raises(ValueError):
        TimeWindowSplitter(data, index_to_info, window_size=0)
        
    with pytest.raises(ValueError):
        TimeWindowSplitter(data, index_to_info, window_size=5)  # Not divisible

def test_invalid_split_ratios():
    data = torch.randn(100, 10)
    
    with pytest.raises(ValueError):
        DataSplitter(data, val_split=0.5, test_split=0.6)  # Sum > 1