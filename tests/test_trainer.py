"""Test MortalTrainer with sandbox reward and function reward environments.

Tests:
1-6: Unit tests for reward environments (sandbox, function, mixed)
7: Full training with sandbox reward (CodeExecutionEnvironment)
8: Full training with function reward (FunctionCodeExecutionEnvironment)

Usage:
    # All tests (unit + training)
    python tests/test_trainer.py

    # Unit tests only (fast, no GPU)
    python tests/test_trainer.py --unit-only

    # Training tests only
    python tests/test_trainer.py --train-only
"""

import sys

import modal
from datasets import Dataset

from mortal.rewards import RewardEnvironment, SandboxConfig
from mortal.rewards.examples import CodeExecutionEnvironment


# --- Custom environment that uses Modal Functions (TRAINING_IMAGE) ---

class FunctionCodeExecutionEnvironment(RewardEnvironment):
    """Like CodeExecutionEnvironment but uses pre-warmed Modal Functions.

    Faster than sandboxes (no container spin-up), runs on TRAINING_IMAGE
    which has torch/trl/numpy/transformers pre-installed.
    """

    name = "FunctionCodeExecution"

    def __init__(self, test_key: str = "testcases", weight: float = 1.0):
        self.test_key = test_key
        self.weight = weight

    def _extract_code(self, completion: str) -> str:
        from mortal.rewards.utils import extract_code_from_completion
        return extract_code_from_completion(completion)

    def _build_test_code(self, code: str, testcases: list[str]) -> str:
        return f"{code}\n\n" + "\n".join(testcases)

    def score(self, completion: str, prompt: str, **kwargs) -> float:
        testcases = kwargs.get(self.test_key, [])
        code = self._extract_code(completion)
        if not testcases:
            return 0.0
        full_code = self._build_test_code(code, testcases)
        result = self.execute_in_function(full_code)
        return 1.0 if result.success else 0.0

    def score_batch(self, completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        all_testcases = kwargs.get(self.test_key, [[] for _ in completions])
        code_strings = []
        for comp, tc in zip(completions, all_testcases):
            code = self._extract_code(comp)
            if tc:
                code_strings.append(self._build_test_code(code, tc))
            else:
                code_strings.append("")

        non_empty = [(i, cs) for i, cs in enumerate(code_strings) if cs]
        if not non_empty:
            return [0.0] * len(completions)

        batch_codes = [cs for _, cs in non_empty]
        batch_results = self.execute_batch_in_function(batch_codes)
        results_map = {}
        for (idx, _), result in zip(non_empty, batch_results):
            results_map[idx] = 1.0 if result.success else 0.0

        return [results_map.get(i, 0.0) for i in range(len(completions))]


# --- Test data ---

COMPLETIONS_PASS = [
    "```python\ndef add(a, b):\n    return a + b\n```",
    "```python\ndef multiply(a, b):\n    return a * b\n```",
]

COMPLETIONS_FAIL = [
    "```python\ndef add(a, b):\n    return a - b\n```",  # wrong
    "```python\ndef multiply(a, b):\n    return a + b\n```",  # wrong
]

TESTCASES = [
    ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
    ["assert multiply(2, 3) == 6", "assert multiply(0, 5) == 0"],
]

PROMPTS = ["Write an add function", "Write a multiply function"]


def make_train_dataset() -> Dataset:
    """Create a small dataset for training tests."""
    return Dataset.from_dict({
        "prompt": [
            "Write a Python function called `add` that takes two numbers and returns their sum.",
            "Write a Python function called `double` that takes a number and returns it multiplied by 2.",
            "Write a Python function called `is_even` that returns True if a number is even.",
            "Write a Python function called `greet` that takes a name and returns 'Hello, <name>!'.",
        ],
        "testcases": [
            ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"],
            ["assert double(5) == 10", "assert double(0) == 0", "assert double(-3) == -6"],
            ["assert is_even(4) == True", "assert is_even(3) == False", "assert is_even(0) == True"],
            ["assert greet('World') == 'Hello, World!'", "assert greet('Alice') == 'Hello, Alice!'"],
        ],
    })


# =====================================================================
# Unit tests (reward environments only, no GPU workers)
# =====================================================================

def test_sandbox_reward():
    """Test CodeExecutionEnvironment (sandbox-based)."""
    print("\n=== Test 1: Sandbox Reward (CodeExecutionEnvironment) ===")
    env = CodeExecutionEnvironment()

    scores = env.score_batch(COMPLETIONS_PASS, PROMPTS, testcases=TESTCASES)
    print(f"  Passing code scores: {scores}")
    assert scores == [1.0, 1.0], f"Expected [1.0, 1.0], got {scores}"

    scores = env.score_batch(COMPLETIONS_FAIL, PROMPTS, testcases=TESTCASES)
    print(f"  Failing code scores: {scores}")
    assert scores == [0.0, 0.0], f"Expected [0.0, 0.0], got {scores}"

    print("  PASSED")


def test_sandbox_reward_with_custom_image():
    """Test CodeExecutionEnvironment with a custom Modal image."""
    print("\n=== Test 2: Sandbox Reward (custom image with numpy) ===")
    env = CodeExecutionEnvironment(
        sandbox_cfg=SandboxConfig(
            image=modal.Image.debian_slim().pip_install("numpy"),
            timeout=60,
        ),
    )

    completions = [
        "```python\nimport numpy as np\ndef dot_product(a, b):\n    return float(np.dot(a, b))\n```",
    ]
    testcases = [["assert dot_product([1, 2, 3], [4, 5, 6]) == 32"]]

    scores = env.score_batch(completions, ["Compute dot product"], testcases=testcases)
    print(f"  Numpy dot product score: {scores}")
    assert scores == [1.0], f"Expected [1.0], got {scores}"

    print("  PASSED")


def test_function_reward():
    """Test FunctionCodeExecutionEnvironment (pre-warmed Modal Function)."""
    print("\n=== Test 3: Function Reward (TRAINING_IMAGE) ===")
    env = FunctionCodeExecutionEnvironment()

    scores = env.score_batch(COMPLETIONS_PASS, PROMPTS, testcases=TESTCASES)
    print(f"  Passing code scores: {scores}")
    assert scores == [1.0, 1.0], f"Expected [1.0, 1.0], got {scores}"

    scores = env.score_batch(COMPLETIONS_FAIL, PROMPTS, testcases=TESTCASES)
    print(f"  Failing code scores: {scores}")
    assert scores == [0.0, 0.0], f"Expected [0.0, 0.0], got {scores}"

    print("  PASSED")


def test_function_reward_with_torch():
    """Test that Function execution has torch available (TRAINING_IMAGE)."""
    print("\n=== Test 4: Function Reward (torch available on TRAINING_IMAGE) ===")
    env = FunctionCodeExecutionEnvironment()

    completions = [
        "```python\nimport torch\ndef tensor_sum(values):\n    return float(torch.tensor(values).sum())\n```",
    ]
    testcases = [["assert tensor_sum([1.0, 2.0, 3.0]) == 6.0"]]

    scores = env.score_batch(completions, ["Compute tensor sum"], testcases=testcases)
    print(f"  Torch tensor sum score: {scores}")
    assert scores == [1.0], f"Expected [1.0], got {scores}"

    print("  PASSED")


def test_mixed_rewards():
    """Test combining sandbox env + callable with weights via dispatch."""
    print("\n=== Test 5: Mixed Rewards (sandbox env + callable, weighted) ===")
    from mortal.rewards import compute_rewards

    def length_reward(completions, **kwargs):
        return [min(len(c) / 200, 1.0) for c in completions]

    sandbox_env = CodeExecutionEnvironment()

    scores = compute_rewards(
        reward_funcs=[sandbox_env, length_reward],
        completions=COMPLETIONS_PASS,
        prompts=PROMPTS,
        weights=[0.7, 0.3],
        testcases=TESTCASES,
    )
    print(f"  Mixed scores (pass): {scores}")
    for s in scores:
        assert 0.5 < s <= 1.0, f"Expected score in (0.5, 1.0], got {s}"

    print("  PASSED")


def test_sandbox_vs_function_same_result():
    """Verify sandbox and function execution produce the same scores."""
    print("\n=== Test 6: Sandbox vs Function (same results) ===")
    sandbox_env = CodeExecutionEnvironment()
    function_env = FunctionCodeExecutionEnvironment()

    sandbox_scores = sandbox_env.score_batch(COMPLETIONS_PASS, PROMPTS, testcases=TESTCASES)
    function_scores = function_env.score_batch(COMPLETIONS_PASS, PROMPTS, testcases=TESTCASES)
    print(f"  Sandbox scores: {sandbox_scores}")
    print(f"  Function scores: {function_scores}")
    assert sandbox_scores == function_scores, (
        f"Mismatch: sandbox={sandbox_scores}, function={function_scores}"
    )

    print("  PASSED")


# =====================================================================
# Training tests (full loop with GPU workers)
# =====================================================================

def test_train_with_sandbox_reward():
    """Full training loop with CodeExecutionEnvironment (sandbox reward)."""
    print("\n=== Test 7: Train with Sandbox Reward (CodeExecutionEnvironment) ===")
    from mortal.trainer import MortalTrainer

    ds = make_train_dataset()
    env = CodeExecutionEnvironment()

    trainer = MortalTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=env,
        train_dataset=ds,
        max_steps=1,
        batch_size=2,
        max_samples=4,
        num_generations=2,
        num_rollout_workers=1,
    )
    trainer.train()

    print("  PASSED")


def test_train_with_function_reward():
    """Full training loop with FunctionCodeExecutionEnvironment (function reward)."""
    print("\n=== Test 8: Train with Function Reward (FunctionCodeExecutionEnvironment) ===")
    from mortal.trainer import MortalTrainer

    ds = make_train_dataset()
    env = FunctionCodeExecutionEnvironment()

    trainer = MortalTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=env,
        train_dataset=ds,
        max_steps=1,
        batch_size=2,
        max_samples=4,
        num_generations=2,
        num_rollout_workers=1,
    )
    trainer.train()

    print("  PASSED")


# =====================================================================
# Main
# =====================================================================

def run_unit_tests():
    test_sandbox_reward()
    test_sandbox_reward_with_custom_image()
    test_function_reward()
    test_function_reward_with_torch()
    test_mixed_rewards()
    test_sandbox_vs_function_same_result()


def run_train_tests():
    test_train_with_sandbox_reward()
    test_train_with_function_reward()


def main():
    args = sys.argv[1:]
    unit_only = "--unit-only" in args
    train_only = "--train-only" in args

    if train_only:
        # Training tests manage their own app.run() via MortalTrainer
        run_train_tests()
        print("\n=== ALL TRAINING TESTS PASSED ===")
        return

    # Unit tests need an app.run() context
    from mortal.app import app
    import mortal.rewards.function_executor  # noqa: F401  — hydrate before app.run()

    with modal.enable_output():
        with app.run():
            run_unit_tests()

    passed = 6

    if not unit_only:
        run_train_tests()
        passed += 2

    print(f"\n=== ALL {passed} TESTS PASSED ===")


if __name__ == "__main__":
    main()
