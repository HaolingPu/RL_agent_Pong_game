# test_cases/test_deploy_agent.py

import torch
from test_cases.test_utils import (
    compare_tensors,
    run_test,
    run_deploy_agent,
    DummyAgent,
    DummyEnv
)


def test_deploy_agent_core():
    agent = DummyAgent(action_to_return=1)
    env = DummyEnv(max_steps=3)
    
    states, actions, rewards, new_states, terminated = run_deploy_agent(agent, env)

    
    expected_states = torch.tensor(
        [
            [  # N = 1
                [0.0, 0.0],  # s0
                [0.0, 1.0],  # s1
                [1.0, 2.0],  # s2
            ]
        ]
    )

    expected_new_states = torch.tensor(
        [
            [
                [0.0, 1.0],  # s1
                [1.0, 2.0],  # s2
                [2.0, 3.0],  # s3
            ]
        ]
    )

    expected_actions = torch.tensor([[[1], [1], [1]]], dtype=torch.int64)  # (1, T, 1)
    expected_rewards = torch.tensor([[[0.0], [1.0], [2.0]]], dtype=torch.float32)
    expected_terminated = torch.tensor(
        [[[False], [False], [True]]], dtype=torch.bool
    )

    
    assert states.shape == expected_states.shape == (1, 3, 2)
    assert new_states.shape == expected_new_states.shape == (1, 3, 2)
    assert actions.shape == expected_actions.shape == (1, 3, 1)
    assert rewards.shape == expected_rewards.shape == (1, 3, 1)
    assert terminated.shape == expected_terminated.shape == (1, 3, 1)

    
    compare_tensors(states, expected_states, label="deploy_agent states")
    compare_tensors(new_states, expected_new_states, label="deploy_agent new_states")
    compare_tensors(actions, expected_actions, label="deploy_agent actions")
    compare_tensors(rewards, expected_rewards, label="deploy_agent rewards")
    compare_tensors(terminated, expected_terminated, label="deploy_agent terminated")


def test_deploy_agent():
    """Wrapper to follow the standard test output format."""
    run_test(test_deploy_agent_core, "deploy_agent")