# test_cases/test_utils.py
"""
Utility functions for testing A2C implementations.
"""
import torch
from torch import nn
import os
import numpy as np
from agent import Policy, Value, Agent, deploy_agent

# test data path
POLICY_TEST_DATA_PATH = "test_data/policy_test.pt"
VALUE_TEST_DATA_PATH = "test_data/value_test.pt"
GET_ACTION_TEST_DATA_PATH = "test_data/get_action_test.pt"
N_STEP_RETURNS_TEST_DATA_PATH = "test_data/n_step_returns_test.pt"
VALUE_LOSS_TEST_DATA_PATH = "test_data/value_loss_test.pt"
POLICY_LOSS_TEST_DATA_PATH = "test_data/policy_loss_test.pt"

def compare_tensors(
    actual: torch.Tensor, expected: torch.Tensor, label: str = "", atol: float = 1e-5
):
    """
    Compare two tensors and raise a ValueError with descriptive stats if they differ.
    Stats reported:
      - shape mismatch
      - maximum absolute difference
      - index of the maximum difference
      - mean absolute difference
      - L2 norm of the difference
      - preview of both tensors (first few elements)
    """
    # First check shape
    if actual.shape != expected.shape:
        raise ValueError(
            f"{label} shape mismatch! Expected {list(expected.shape)}, "
            f"got {list(actual.shape)}"
        )

    # Then check if close within the given tolerance
    if not torch.allclose(actual, expected, atol=atol):
        diff = actual - expected
        abs_diff = diff.abs()
        max_diff = abs_diff.max().item()
        max_diff_idx = torch.argmax(abs_diff).item()
        flat_actual = actual.view(-1)
        flat_expected = expected.view(-1)

        # Also compute mean absolute difference and L2 distance
        mean_diff = abs_diff.mean().item()
        l2_diff = torch.norm(diff).item()
        
        # Add a preview of both tensors (first few elements or slice around the max difference)
        # For 1D tensors, show elements around the max difference
        preview_size = 5  # Number of elements to show
        
        # Get tensor previews
        if len(actual.shape) == 1:
            # For 1D tensors, show elements around max_diff_idx
            start_idx = max(0, max_diff_idx - preview_size // 2)
            end_idx = min(len(flat_actual), start_idx + preview_size)
            actual_preview = flat_actual[start_idx:end_idx]
            expected_preview = flat_expected[start_idx:end_idx]
            preview_range = f"[{start_idx}:{end_idx}]"
        else:
            # For higher dimensional tensors, show the first few elements
            flat_preview_size = min(preview_size, flat_actual.numel())
            actual_preview = flat_actual[:flat_preview_size]
            expected_preview = flat_expected[:flat_preview_size]
            preview_range = f"[0:{flat_preview_size}]"
            
            # Also try to show the region with the max difference
            idx_tuple = np.unravel_index(max_diff_idx, actual.shape)
            region_info = f"\n  Region with max difference (index {idx_tuple}):"
            
            # Try to get a small slice around the max difference area
            slice_indices = []
            for i, dim_size in enumerate(actual.shape):
                idx = idx_tuple[i]
                start = max(0, idx - 1)
                end = min(dim_size, idx + 2)
                slice_indices.append(slice(start, end))
            
            # Create the slices for displaying the region with max difference
            try:
                actual_region = actual[tuple(slice_indices)]
                expected_region = expected[tuple(slice_indices)]
                region_preview = (
                    f"\n  Actual region:\n{actual_region}\n"
                    f"  Expected region:\n{expected_region}"
                )
                region_info += region_preview
            except:
                # If we couldn't extract a region (e.g., for complex shapes), skip this
                region_info = ""

        error_msg = (
            f"{label} values mismatch!\n"
            f"  Maximum absolute difference: {max_diff:.6f} (at index {max_diff_idx})\n"
            f"  Expected vs. got at that index: "
            f"{flat_expected[max_diff_idx]:.6f} vs. {flat_actual[max_diff_idx]:.6f}\n"
            f"  Mean absolute difference: {mean_diff:.6f}\n"
            f"  L2 norm of the difference: {l2_diff:.6f}\n"
            f"  Preview {preview_range}:\n"
            f"    Expected: {expected_preview.tolist()}\n"
            f"    Actual:   {actual_preview.tolist()}"
        )
        
        if 'region_info' in locals() and region_info:
            error_msg += region_info
            
        error_msg += f"\n  (allclose with atol={atol} failed)\n"
        
        raise ValueError(error_msg)
    
def load_test_data(test_data_path):
    """Load and validate test data."""
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file '{test_data_path}' not found.")
    return torch.load(test_data_path)

def run_test(test_func, name):
    """Run a test function with standard output formatting."""
    print(f"Testing {name}...", end="")
    test_func()
    print(" Passed!")

def setup_policy_with_test_data(test_data):
    """Configure an Policy with weights from test data."""
    # Create policy with test parameters
    state_space = test_data["state_space"]
    action_space = test_data["action_space"]
    hidden_dim = test_data["hidden_dim"]
    
    policy = Policy(state_space = state_space, action_space = action_space, hidden_dim = hidden_dim)
    
    # Set policy weights
    policy.load_state_dict(test_data["state_dict"])
    
    return policy

def setup_value_with_test_data(test_data):
    """Configure an Value with weights from test data."""
    # Create value with test parameters
    state_space = test_data["state_space"]    
    hidden_dim = test_data["hidden_dim"]
    
    value = Value(state_space = state_space, hidden_dim = hidden_dim)
    
    # Set value weights
    value.load_state_dict(test_data["state_dict"])
    
    return value

def setup_agent_from_test_data(test_data):
    """Reconstruct the Agent exactly as it was during test data generation"""
    agent = Agent(
        state_space=test_data["state_space"],
        action_space=test_data["action_space"],
        gamma=test_data["gamma"],
        lr=test_data["lr"],
        max_training_steps=test_data["max_training_steps"],
    )

    if "policy_state_dict" in test_data:
        agent.policy.load_state_dict(test_data["policy_state_dict"])

    if "value_state_dict" in test_data:
        agent.value.load_state_dict(test_data["value_state_dict"])

        
    return agent

def run_deploy_agent(agent, env):
    return deploy_agent(agent, env)

class DummyAgent:
    """
    Minimal agent with a deterministic get_action.    
    """
    def __init__(self, action_to_return: int = 0):
        self.action_to_return = action_to_return

    def get_action(self, state: torch.Tensor) -> int:
        # Ignore state, always return the same action for determinism
        return self.action_to_return


class DummyEnv:
    """
    Minimal environment with deterministic dynamics:
    - reset() -> state = [0., 0.]
    - step(action) -> state = [t, t+1], reward = float(t), terminated when t == max_steps-1
    """
    def __init__(self, max_steps: int = 3):
        self.max_steps = max_steps
        self.t = 0  # internal time step

    def reset(self, seed=None):
        self.t = 0
        # First state s0
        return torch.tensor([0.0, 0.0])

    def step(self, action: int):
        """
        Return:
          new_state: [t, t+1]
          reward: float(t)
          terminated: True when t == max_steps - 1
          truncated: always False here
        """
        new_state = torch.tensor([float(self.t), float(self.t + 1)])
        reward = float(self.t)
        terminated = (self.t == self.max_steps - 1)
        truncated = False
        self.t += 1
        return new_state, reward, terminated, truncated