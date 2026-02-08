"""Modal GRPO - Serverless GRPO training with TRL and vLLM.

This package provides a veRL-style serverless architecture for GRPO training:
- Local TRL installation
- Single unified volume for all storage
- NVIDIA CUDA-based images
- Separate workers: Actor (TRL GRPOTrainer), Rollout (standalone vLLM), Reward (Sandboxes)

Usage:
    # Create volume first (one-time)
    modal volume create grpo-trl-storage

    # Run training
    modal run MRL/train.py --model "Qwen/Qwen2-0.5B-Instruct" --epochs 5

    # Run with detach (background)
    modal run --detach MRL/train.py
"""

from MRL.app import app, volume, TRAINING_IMAGE, VLLM_IMAGE
from MRL.config import OrchestratorConfig, ModelConfig, TrainingConfig, GenerationConfig
from MRL.trainer import MRLTrainer

__all__ = [
    "app",
    "volume",
    "TRAINING_IMAGE",
    "VLLM_IMAGE",
    "OrchestratorConfig",
    "ModelConfig",
    "TrainingConfig",
    "GenerationConfig",
    "MRLTrainer",
]
