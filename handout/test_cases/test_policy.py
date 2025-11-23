# test_cases/test_policy.py

import torch
from test_cases.test_utils import (
    load_test_data, 
    compare_tensors,
    setup_policy_with_test_data,
    run_test,
    POLICY_TEST_DATA_PATH
    )

def test_policy_foward():    
    """Test the complete Policy forward pass implementation."""
    test_data = load_test_data(POLICY_TEST_DATA_PATH)    
    policy = setup_policy_with_test_data(test_data)
    policy.eval()

    x = test_data["input"]
    y_expected = test_data["output"]
    
    # Forward pass on the stored input
    with torch.no_grad():
        y_actual = policy(x)

    # Compare expected vs actual
    compare_tensors(
        y_actual, 
        y_expected, 
        label="Policy foward",
        atol=1e-5)

def test_policy_model():
    """Run all Policy test."""
    run_test(test_policy_foward, "Policy model forward pass")