# test_cases/test_n_step_returns.py

import torch
from test_cases.test_utils import (
    load_test_data, 
    compare_tensors, 
    run_test,
    setup_agent_from_test_data,
    N_STEP_RETURNS_TEST_DATA_PATH
    )    



def test_n_step_returns_core():
    """
    Test that n_step_returns() matches the reference implementation
    for a small batch of trajectories, including termination handling.
    """
    test_data = load_test_data(N_STEP_RETURNS_TEST_DATA_PATH)
    agent = setup_agent_from_test_data(test_data)
    
    n = int(test_data["n"])
    rewards = test_data["rewards"]               # (N, T, 1)
    next_state_values = test_data["next_state_values"]  # (N, T, 1)
    terminated = test_data["terminated"]         # (N, T, 1)
    expected_returns = test_data["expected_returns"]
    
    assert rewards.dim() == 3, "rewards must have shape (N, T, 1)"
    assert next_state_values.shape == rewards.shape, "next_state_values must match rewards shape"
    assert terminated.shape == rewards.shape, "terminated must match rewards shape"
    
    actual_returns = agent.n_step_returns(n, rewards, next_state_values, terminated)
    
    compare_tensors(
        actual_returns,
        expected_returns,
        label="n_step_returns output",
        atol=1e-5,
    )


def test_n_step_returns():
    """Wrapper for formatted output."""
    run_test(test_n_step_returns_core, "n_step_returns")