"""Test MRLTrainer with a custom reward function (local orchestrator path).

Usage:
    python tests/test_trainer.py
"""

from MRL.trainer import MRLTrainer


def length_reward(completions, **kwargs):
    """Simple reward: longer completions score higher (capped at 1.0)."""
    return [min(len(c) / 100, 1.0) for c in completions]


def main():
    trainer = MRLTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=length_reward,
        max_steps=2,
        batch_size=2,
        max_samples=8,
        num_generations=2,
        num_rollout_workers=1,
    )
    trainer.train()


if __name__ == "__main__":
    main()
