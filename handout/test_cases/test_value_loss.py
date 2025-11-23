# test_cases/test_value_loss.py

import torch
from test_cases.test_utils import (
    load_test_data,
    compare_tensors,
    run_test,
    setup_agent_from_test_data,
    VALUE_LOSS_TEST_DATA_PATH
    )


def test_value_loss_core():
    """
    Test that value_loss() matches the reference implementation given fixed
    states, rewards, next_states, terminated, and value network weights.
    """
    test_data = load_test_data(VALUE_LOSS_TEST_DATA_PATH)
    agent = setup_agent_from_test_data(test_data)

    states = test_data["states"]          # (N, T, S)
    next_states = test_data["next_states"]
    rewards = test_data["rewards"]        # (N, T, 1)
    terminated = test_data["terminated"]  # (N, T, 1)
    expected_loss = test_data["expected_loss"]  # scalar tensor
    
    assert states.dim() == 3, "states must have shape (N, T, S)"
    assert next_states.shape == states.shape, "next_states must match states shape"
    assert rewards.shape[:2] == states.shape[:2] and rewards.shape[-1] == 1, \
        "rewards must have shape (N, T, 1)"
    assert terminated.shape == rewards.shape, "terminated must match rewards shape"
    
    actual_loss = agent.value_loss(states, rewards, next_states, terminated)

    
    assert actual_loss.dim() == 0, "value_loss must return a scalar tensor"
    
    # Compare to reference loss
    compare_tensors(
        actual_loss,
        expected_loss,
        label="Value loss",
        atol=1e-5,
    )


def test_value_loss():
    """Wrapper to match the standard formatted test output."""
    run_test(test_value_loss_core, "value_loss")