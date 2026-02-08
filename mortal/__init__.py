"""MORTAL - Modal Orchestrated Reinforcement Training Architecture for LLMs.

This package provides a veRL-style serverless architecture for GRPO training:
- Single unified volume for all storage
- NVIDIA CUDA-based images
- Separate workers: Actor (TRL GRPOTrainer), Rollout (standalone vLLM), Reward

Usage:
    from mortal import MortalTrainer

    trainer = MortalTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=my_reward_fn,
        train_dataset=dataset,
        max_steps=5,
    )
    trainer.train()
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from mortal.app import app, volume, TRAINING_IMAGE, VLLM_IMAGE
from mortal.config import OrchestratorConfig, ModelConfig, TrainingConfig, GenerationConfig
from mortal.trainer import MortalTrainer
from mortal.rewards.base import RewardEnvironment, SandboxConfig, FunctionConfig, ExecutionResult

__all__ = [
    "app",
    "volume",
    "TRAINING_IMAGE",
    "VLLM_IMAGE",
    "OrchestratorConfig",
    "ModelConfig",
    "TrainingConfig",
    "GenerationConfig",
    "MortalTrainer",
    "RewardEnvironment",
    "SandboxConfig",
    "FunctionConfig",
    "ExecutionResult",
]
