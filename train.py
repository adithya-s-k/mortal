"""Entry point for Modal GRPO training."""

import modal

from MRL.app import app, volume
from MRL.orchestrator import train, train_simple


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2-0.5B-Instruct",
    epochs: int = 5,
    max_steps: int = 5,
    batch_size: int = 8,
    num_rollout_workers: int = 2,
    num_generations: int = 4,
    max_samples: int = 128,
    max_tokens: int = 8000,
    max_model_len: int = 16384,
    learning_rate: float = 5e-6,
    save_steps: int = 100,
    sync_weights_every: int = 1,
    weight_sync_method: str = "direct",
    simple_mode: bool = False,
):
    """Launch GRPO training on Modal.

    Args:
        model: Model name or path (default: Qwen/Qwen2-0.5B-Instruct)
        epochs: Number of training epochs (default: 5)
        max_steps: Maximum training steps, -1 for unlimited (default: 5 for testing)
        batch_size: Batch size (default: 8)
        num_rollout_workers: Number of vLLM rollout workers (default: 2)
        num_generations: Generations per prompt for GRPO (default: 4)
        max_samples: Maximum dataset samples, 0 for full dataset (default: 128)
        max_tokens: Maximum tokens to generate per completion (default: 8000)
        max_model_len: Maximum model context length (default: 16384)
        learning_rate: Learning rate (default: 5e-6)
        save_steps: Save checkpoint every N steps (default: 100)
        sync_weights_every: Sync weights to rollout workers every N steps (default: 1)
        weight_sync_method: Method for syncing weights:
            "reload" (recommended) - uses vLLM v1 sleep/wake_up/reload_weights
            "volume" - saves to shared volume, workers reload
            "direct" - in-memory transfer (limited vLLM support)
            "checkpoint" - full checkpoint save + reload (slowest)
        simple_mode: Use TRL's built-in training loop instead of orchestrator (default: False)
    """
    config = {
        "model_name": model,
        "num_epochs": epochs,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "num_rollout_workers": num_rollout_workers,
        "num_generations": num_generations,
        "max_samples": max_samples if max_samples > 0 else None,
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "learning_rate": learning_rate,
        "save_steps": save_steps,
        "sync_weights_every": sync_weights_every,
        "weight_sync_method": weight_sync_method,
    }

    print("Starting GRPO training with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    if simple_mode:
        print("\nUsing simple mode (TRL built-in training loop)")
        result = train_simple.remote(config)
    else:
        print("\nUsing orchestrator mode (veRL-style distributed training)")
        result = train.remote(config)

    print(f"\nTraining result: {result}")


# Separate test functions that can be run directly via modal run
@app.function()
def test_rollout_fn(
    model: str = "Qwen/Qwen2-0.5B-Instruct",
    prompt: str = "Write a Python function to check if a number is prime.",
    max_tokens: int = 512,
):
    """Test rollout worker generation."""
    from MRL.workers.rollout import RolloutWorker

    worker = RolloutWorker()

    print(f"Testing rollout with model: {model}")
    print(f"Prompt: {prompt}")

    result = worker.generate.remote(
        prompts=[prompt],
        model_path=model,
        max_tokens=max_tokens,
        temperature=0.7,
        n=2,  # Generate 2 completions
    )

    print("\nCompletions:")
    for i, completion in enumerate(result["completions"]):
        print(f"\n--- Completion {i + 1} ---")
        print(completion[:500] + "..." if len(completion) > 500 else completion)

    return result


@app.function()
def test_reward_fn():
    """Test reward computation with a simple example."""
    from MRL.workers.reward import compute_reward

    # Test case: simple function that should pass
    completion_good = """```python
def add(a, b):
    return a + b
```"""
    testcase_good = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]

    # Test case: function that should fail
    completion_bad = """```python
def add(a, b):
    return a - b  # Wrong!
```"""
    testcase_bad = ["assert add(1, 2) == 3"]

    print("Testing reward computation...")

    reward_good = compute_reward.remote(completion_good, testcase_good)
    print(f"Good completion reward: {reward_good} (expected: 1)")

    reward_bad = compute_reward.remote(completion_bad, testcase_bad)
    print(f"Bad completion reward: {reward_bad} (expected: 0)")

    return {"good": reward_good, "bad": reward_bad}


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={"/storage": volume},
)
def list_checkpoints_fn():
    """List available checkpoints in the volume."""
    import os

    checkpoint_dir = "/storage/checkpoints"
    print(f"Listing checkpoints in {checkpoint_dir}:")

    if not os.path.exists(checkpoint_dir):
        print("  (no checkpoints directory found)")
        return []

    checkpoints = []
    for item in sorted(os.listdir(checkpoint_dir)):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path):
            checkpoints.append(item)
            print(f"  - {item}")

    if not checkpoints:
        print("  (no checkpoints found)")

    return checkpoints
