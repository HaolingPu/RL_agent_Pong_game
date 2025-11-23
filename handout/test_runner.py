"""
Main test runner for A2C implementation.
This file orchestrates the tests but delegates the actual test logic to separate modules.
"""
import warnings
import os
from test_cases.test_policy import test_policy_model
from test_cases.test_value import test_value_model
from test_cases.test_get_action import test_get_action
from test_cases.test_n_step_returns import test_n_step_returns
from test_cases.test_value_loss import test_value_loss
from test_cases.test_policy_loss import test_policy_loss
from test_cases.test_deploy_agent import test_deploy_agent

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

def run_tests():
    """Run all tests for the A2C implementation."""

    tests = [
        ("Policy", test_policy_model),
        ("Value", test_value_model),
        ("Get Action", test_get_action),
        ("N Step Return", test_n_step_returns),
        ("Value Loss", test_value_loss),
        ("Policy Loss", test_policy_loss),
        ("Deploy Agent", test_deploy_agent),
    ]

    passed = 0
    total = len(tests)

    print("\n===== Starting A2C Implementation Tests =====\n")
    print(
        "IMPORTANT NOTE: Tests are independent. If a later test passes but an earlier test fails,"
    )
    print(
        "it does NOT mean the earlier component is correct."
    )
    
    print("You must pass ALL tests for your implementation to be fully correct.\n")

    for test_name, test_func in tests:
        print(f"Running test: {test_name}...")
        try:
            test_func()
            passed += 1
            print(f"{test_name}: ✓ PASSED\n")
        except Exception as e:
            print(f" ✗ FAILED")
            print(f"Error details:\n{str(e)}\n")

    print("===== Test Results Summary =====")
    print(f"Passed: {passed}/{total} tests")

    if passed == total:
        print("🎉 All tests passed! Your implementation is correct.")
    else:
        print(f"❌ {total - passed} test(s) failed. Please review your implementation.")
        print(
            "Tip: Look at the detailed error messages above to see how far off your outputs were."
        )
        print(
            "Remember: Tests are independent, so passing a later test doesn't guarantee correctness of earlier components."
        )

    print("\n====================================")


if __name__ == "__main__":
    run_tests()