# test_cases/test_value.py
import torch
from test_cases.test_utils import (
    load_test_data, 
    compare_tensors,
    setup_value_with_test_data,
    run_test,
    VALUE_TEST_DATA_PATH
    )

def test_value_foward():    
    """Test the complete Value forward pass implementation."""
    test_data = load_test_data(VALUE_TEST_DATA_PATH)    
    value = setup_value_with_test_data(test_data)
    value.eval()

    x = test_data["input"]
    y_expected = test_data["output"]    
    
    # Forward pass on the stored input
    with torch.no_grad():
        y_actual = value(x)

    # Compare expected vs actual
    compare_tensors(
        y_actual, 
        y_expected, 
        label="Value foward",
        atol=1e-5)

def test_value_model():
    """Run all Value test."""
    run_test(test_value_foward, "Value model forward pass")