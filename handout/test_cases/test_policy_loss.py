# test_cases/test_policy_loss.py

import torch
from test_cases.test_utils import (
    load_test_data, 
    compare_tensors, 
    run_test,
    setup_agent_from_test_data,
    POLICY_LOSS_TEST_DATA_PATH
    )

def test_policy_loss_core():
    """
    Test that policy_loss() matches the reference implementation given fixed
    states, actions, rewards, next_states, terminated, and network weights.
    """
    test_data = load_test_data(POLICY_LOSS_TEST_DATA_PATH)
    agent = setup_agent_from_test_data(test_data)

    states = test_data["states"]          # (N, T, S)
    next_states = test_data["next_states"]
    actions = test_data["actions"]        # (N, T, 1), int64
    rewards = test_data["rewards"]        # (N, T, 1)
    terminated = test_data["terminated"]  # (N, T, 1)
    expected_loss = test_data["expected_loss"]  # scalar tensor

    assert states.dim() == 3, "states must have shape (N, T, S)"
    assert next_states.shape == states.shape, "next_states must match states shape"
    assert rewards.shape[:2] == states.shape[:2] and rewards.shape[-1] == 1, \
        "rewards must have shape (N, T, 1)"
    assert terminated.shape == rewards.shape, "terminated must match rewards shape"
    assert actions.shape[:2] == states.shape[:2] and actions.shape[-1] == 1, \
        "actions must have shape (N, T, 1)"
    
    actual_loss = agent.policy_loss(states, actions, rewards, next_states, terminated)

    assert actual_loss.dim() == 0, "policy_loss must return a scalar tensor"

    compare_tensors(
        actual_loss,
        expected_loss,
        label="policy_loss",
        atol=1e-5,
    )


def test_policy_loss():
    """Wrapper to match the standard formatted test output."""
    run_test(test_policy_loss_core, "policy_loss")