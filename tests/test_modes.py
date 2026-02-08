"""Test all execution modes: SingleNode and Distributed.

Tests:
1. single_node_basic        — SingleNode, no vLLM, sandbox rewards
2. single_node_local_reward  — SingleNode, no vLLM, local rewards
3. single_node_vllm_colocate — SingleNode, vLLM colocate, sandbox rewards
4. single_node_vllm_serve    — SingleNode, vLLM serve (2xA100-80GB), sandbox rewards
5. distributed_basic         — Distributed, sandbox rewards
6. distributed_custom_reward — Distributed, custom callable reward

Usage:
    # Run a specific test
    python tests/test_modes.py single_node_basic

    # Run multiple tests
    python tests/test_modes.py single_node_basic distributed_basic

    # Run all tests
    python tests/test_modes.py all

    # Run with --detach (fire and forget)
    python tests/test_modes.py single_node_basic --detach

    # List available tests
    python tests/test_modes.py --list
"""

import argparse
import sys

from datasets import load_dataset

from mortal import MortalTrainer, SingleNode, Distributed, GPUConfig


# Shared small test params
SMALL = dict(
    model="Qwen/Qwen2-0.5B-Instruct",
    max_steps=2,
    batch_size=2,
    max_samples=8,
    num_generations=2,
)


def test_single_node_basic(detach=False):
    """SingleNode, no vLLM, sandbox rewards."""
    trainer = MortalTrainer(
        **SMALL,
        mode=SingleNode(gpu="A100"),
    )
    trainer.train(detach=detach)


def test_single_node_local_reward(detach=False):
    """SingleNode, no vLLM, local (in-process) rewards."""
    trainer = MortalTrainer(
        **SMALL,
        mode=SingleNode(gpu="A100", reward_type="local"),
    )
    trainer.train(detach=detach)


def test_single_node_vllm_colocate(detach=False):
    """SingleNode, vLLM colocate (shared GPU memory)."""
    trainer = MortalTrainer(
        **SMALL,
        mode=SingleNode(gpu="A100-80GB", use_vllm=True, vllm_mode="colocate"),
    )
    trainer.train(detach=detach)


def test_single_node_vllm_serve(detach=False):
    """SingleNode, vLLM serve (dedicated GPU). Requires 2xA100-80GB."""
    trainer = MortalTrainer(
        **SMALL,
        mode=SingleNode(
            gpu=GPUConfig("A100-80GB", count=2),
            use_vllm=True,
            vllm_mode="serve",
        ),
    )
    trainer.train(detach=detach)


def test_distributed_basic(detach=False):
    """Distributed (veRL-style), sandbox rewards."""
    trainer = MortalTrainer(
        **SMALL,
        mode=Distributed(actor="A100", rollout="A10G", num_rollout_workers=1),
    )
    trainer.train(detach=detach)


def test_distributed_custom_reward(detach=False):
    """Distributed, custom callable reward (local orchestrator)."""
    def length_reward(completions, **kwargs):
        return [min(len(c) / 500, 1.0) for c in completions]

    dataset = load_dataset(
        "OpenCoder-LLM/opc-sft-stage2", "educational_instruct", split="train"
    )
    dataset = dataset.rename_column("instruction", "prompt")
    dataset = dataset.rename_column("testcase", "testcases")
    dataset = dataset.select(range(8))

    trainer = MortalTrainer(
        **SMALL,
        mode=Distributed(actor="A100", rollout="A10G", num_rollout_workers=1),
        reward_funcs=length_reward,
        train_dataset=dataset,
    )
    trainer.train(detach=detach)


TESTS = {
    "single_node_basic": test_single_node_basic,
    "single_node_local_reward": test_single_node_local_reward,
    "single_node_vllm_colocate": test_single_node_vllm_colocate,
    "single_node_vllm_serve": test_single_node_vllm_serve,
    "distributed_basic": test_distributed_basic,
    "distributed_custom_reward": test_distributed_custom_reward,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MORTAL execution modes")
    parser.add_argument("tests", nargs="*", default=[], help="Test names to run, or 'all'")
    parser.add_argument("--detach", action="store_true", help="Run detached from terminal")
    parser.add_argument("--list", action="store_true", help="List available tests")
    args = parser.parse_args()

    if args.list or not args.tests:
        print("Available tests:")
        for name, fn in TESTS.items():
            print(f"  {name:30s} — {fn.__doc__}")
        sys.exit(0)

    to_run = list(TESTS.keys()) if "all" in args.tests else args.tests

    for name in to_run:
        if name not in TESTS:
            print(f"Unknown test: {name}")
            print(f"Available: {', '.join(TESTS.keys())}")
            sys.exit(1)

    for name in to_run:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}\n")
        TESTS[name](detach=args.detach)
        print(f"\n✓ {name} completed\n")
