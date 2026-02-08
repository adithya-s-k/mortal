"""Minimal example: train with sandbox reward (Modal code execution).

Usage:
    python examples/simple_reward.py
"""

from MRL import MRLTrainer

if __name__ == "__main__":
    trainer = MRLTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="sandbox",
        max_steps=2,
        batch_size=2,
        max_samples=8,
        num_generations=2,
        num_rollout_workers=1,
    )
    trainer.train()
