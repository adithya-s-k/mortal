"""MORTAL - Modal app, images, and volume definitions for GRPO training."""

import modal

# Modal App
app = modal.App("mortal")

# Paths
STORAGE_PATH = "/storage"

# Single unified volume for all storage
volume = modal.Volume.from_name("grpo-trl-storage", create_if_missing=True)

# CUDA versions
CUDA_TRAINING = "12.8.0"
CUDA_VLLM = "12.8.1"

# Training Image (for Actor worker - uses TRL GRPOTrainer)
TRAINING_IMAGE = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_TRAINING}-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("libglib2.0-0", "libgl1", "libglx-mesa0", "libgl1-mesa-dri", "git")
    .run_commands("pip install uv")
    .run_commands(
        "uv pip install 'trl[vllm]' --system",
        "uv pip install wandb datasets accelerate peft bitsandbytes hf_transfer --system",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": f"{STORAGE_PATH}/hf_cache",
    })
    .add_local_python_source("mortal")
)

# vLLM Image (for Rollout workers - standalone vLLM inference)
# Using debian_slim which works better with vLLM multiprocessing
# Note: VLLM_USE_V1=1 (default) enables v1 engine with collective_rpc for direct weight updates
VLLM_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm transformers safetensors hf_transfer --system")
    .env({
        "HF_HOME": f"{STORAGE_PATH}/hf_cache",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
    .add_local_python_source("mortal")
)

# NOTE: Do NOT use `with TRAINING_IMAGE.imports()` to eagerly import torch/trl.
# This would initialize CUDA before CUDA_VISIBLE_DEVICES can be set in serve mode,
# causing GPU device conflicts. Import torch/trl lazily inside methods instead.
