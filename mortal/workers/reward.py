"""Reward computation using Modal Sandboxes for code execution.

NOTE: This module is kept for backward compatibility. For new code, prefer
using the mortal.rewards package (RewardEnvironment, CodeExecutionEnvironment, etc.).
"""

from typing import Iterable, Sequence

import modal

# Import app from the shared app module
from mortal.app import app

STORAGE_PATH = "/storage"


def get_generated_code_and_test_cases(completion: str, testcase: Sequence[str]) -> str:
    """Extract code from completion and combine with test cases.

    The completions are expected to follow the format "```python ...```".
    The test cases are a list of assert statements.

    Args:
        completion: Model completion, potentially with markdown code blocks
        testcase: List of test case assert statements

    Returns:
        Combined code string ready for execution
    """
    if "```python" in completion:
        # Find the start and end of the code block
        start_idx = completion.find("```python") + len("```python")
        end_idx = completion.find("```", start_idx)
        if end_idx != -1:
            code = completion[start_idx:end_idx].strip()
        else:
            code = completion[start_idx:].strip()
    elif "```" in completion:
        # Try generic code block
        start_idx = completion.find("```") + len("```")
        # Skip language identifier if present
        newline_idx = completion.find("\n", start_idx)
        if newline_idx != -1 and newline_idx - start_idx < 20:
            start_idx = newline_idx + 1
        end_idx = completion.find("```", start_idx)
        if end_idx != -1:
            code = completion[start_idx:end_idx].strip()
        else:
            code = completion[start_idx:].strip()
    else:
        code = completion.strip()

    test_cases = "\n".join(testcase)
    full_code = f"{code}\n\n{test_cases}"
    return full_code


@app.function()
def compute_reward(completion: str, testcase: Sequence[str]) -> int:
    """Compute reward for a single completion using Modal Sandbox.

    Executes the generated code with test cases in a secure sandbox.
    Returns 1 if all tests pass, 0 otherwise.

    Args:
        completion: Model completion
        testcase: List of test case assert statements

    Returns:
        Reward score (1 for success, 0 for failure)
    """
    sb = None
    score = 0

    code_to_execute = get_generated_code_and_test_cases(completion, testcase)

    try:
        sb = modal.Sandbox.create(app=app)
        p = sb.exec("python", "-c", code_to_execute, timeout=30)
        p.wait()
        return_code = p.returncode
        if return_code == 0:
            score = 1
    except Exception as e:
        print(f"Sandbox execution error: {e}")
    finally:
        if sb is not None:
            sb.terminate()

    return score


@app.function()
def compute_reward_batch(
    completions: list[str], testcases: list[Sequence[str]]
) -> list[int]:
    """Compute rewards for a batch of completions.

    Args:
        completions: List of completions
        testcases: List of test case lists (one per completion)

    Returns:
        List of reward scores
    """
    return list(compute_reward.starmap(zip(completions, testcases)))


def reward_helper_function(
    completions: Sequence[str], testcases: Sequence[Sequence[str]], **kwargs
) -> Iterable[int]:
    """TRL-compatible reward function using parallel sandbox execution.

    This function conforms to TRL's reward function signature and can be
    passed directly to GRPOTrainer.

    Args:
        completions: Sequence of model completions
        testcases: Sequence of test case lists
        **kwargs: Additional arguments (ignored)

    Returns:
        Iterable of reward scores
    """
    return compute_reward.starmap(zip(completions, testcases))


# Additional reward functions can be defined here


@app.function()
def compute_reward_with_partial_credit(
    completion: str, testcase: Sequence[str]
) -> float:
    """Compute reward with partial credit for passing some tests.

    Args:
        completion: Model completion
        testcase: List of test case assert statements

    Returns:
        Reward score between 0 and 1
    """
    if not testcase:
        return 0.0

    code = get_generated_code_and_test_cases(completion, [])  # Just get the code

    passed = 0
    total = len(testcase)

    for test in testcase:
        sb = None
        try:
            sb = modal.Sandbox.create(app=app)
            test_code = f"{code}\n\n{test}"
            p = sb.exec("python", "-c", test_code, timeout=10)
            p.wait()
            if p.returncode == 0:
                passed += 1
        except Exception:
            pass
        finally:
            if sb is not None:
                sb.terminate()

    return passed / total if total > 0 else 0.0


def partial_credit_reward_function(
    completions: Sequence[str], testcases: Sequence[Sequence[str]], **kwargs
) -> Iterable[float]:
    """TRL-compatible reward function with partial credit.

    Args:
        completions: Sequence of model completions
        testcases: Sequence of test case lists
        **kwargs: Additional arguments (ignored)

    Returns:
        Iterable of reward scores (0 to 1)
    """
    return compute_reward_with_partial_credit.starmap(zip(completions, testcases))
