# test_cases/test_get_action.py

import torch
from test_cases.test_utils import (
    load_test_data,     
    setup_agent_from_test_data,
    run_test,
    GET_ACTION_TEST_DATA_PATH
    )



def test_get_action_deterministic():
    """
    Test that get_action returns the same action as the reference implementation
    given fixed weights, fixed state, and fixed RNG seed.
    """
    
    test_data = load_test_data(GET_ACTION_TEST_DATA_PATH)
    agent = setup_agent_from_test_data(test_data)
    
    state = test_data["state"]
    expected_action = int(test_data["expected_action"])
    
    # DO NOT CHANGE THIS!!!
    torch.manual_seed(10301)
    
    actual_action = agent.get_action(state)
    
    assert isinstance(actual_action, int), (
        "get_action must return a Python int, but got "
        f"{type(actual_action)}."
    )

    assert actual_action == expected_action, (
        f"Incorrect action returned by get_action.\n"
        f"Refer to PyTorch documentation to understand how to sample\n"
        f"from a categorical distribution.\n"
        f"Expected: {expected_action}\n"
        f"Got:      {actual_action}"
    )


def test_get_action():
    """Wrapper to match test output formatting."""
    run_test(test_get_action_deterministic, "get_action")


