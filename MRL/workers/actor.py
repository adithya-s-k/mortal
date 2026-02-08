"""Actor worker for GRPO training using TRL's GRPOTrainer."""

import io
from typing import Optional

import modal

# Import app and resources from the shared app module
# Note: For Modal, all files need to share the same app instance
from MRL.app import app, volume, TRAINING_IMAGE

STORAGE_PATH = "/storage"


@app.cls(
    image=TRAINING_IMAGE,
    gpu="A100",
    volumes={STORAGE_PATH: volume},
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
    timeout=60 * 60 * 24,  # 24 hours
)
class ActorWorker:
    """Training worker using TRL's GRPOTrainer.

    This worker handles the training loop, computing GRPO loss and updating
    model weights. It receives pre-computed completions and rewards from
    the orchestrator.
    """

    @modal.enter()
    def setup(self):
        """Lazy initialization on container start."""
        self.trainer = None
        self.model = None
        self.tokenizer = None
        self.config = None
        self.initialized = False
        print("ActorWorker container started, awaiting initialization...")

    def _do_initialize(self, config: dict, resume_from: Optional[str] = None) -> bool:
        """Internal initialization logic.

        Args:
            config: Configuration dictionary with training parameters
            resume_from: Optional checkpoint path to resume from

        Returns:
            True if initialization successful
        """
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        self.config = config
        model_name = config.get("model_name", "Qwen/Qwen2-0.5B-Instruct")

        print(f"Initializing ActorWorker with model: {model_name}")

        # Load dataset
        dataset = load_dataset(
            config.get("dataset_name", "OpenCoder-LLM/opc-sft-stage2"),
            config.get("dataset_config", "educational_instruct"),
            split=config.get("dataset_split", "train"),
        )
        dataset = dataset.rename_column("instruction", "prompt")
        dataset = dataset.rename_column("testcase", "testcases")

        max_samples = config.get("max_samples")
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        print(f"Dataset loaded with {len(dataset)} samples")

        # Create dummy reward function (actual rewards come from orchestrator)
        def dummy_reward_func(completions, **kwargs):
            return [0.0] * len(completions)

        # Build GRPOConfig kwargs
        grpo_kwargs = {
            "output_dir": config.get("checkpoint_dir", f"{STORAGE_PATH}/checkpoints"),
            "use_vllm": False,  # We handle vLLM separately via RolloutWorkers
            "report_to": "none",  # Disable wandb in actor (orchestrator handles it)
            "per_device_train_batch_size": config.get("batch_size", 8),
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
            "learning_rate": config.get("learning_rate", 5e-6),
            "num_train_epochs": config.get("num_epochs", 5),
            "max_steps": config.get("max_steps", -1),
            "save_steps": config.get("save_steps", 100),
            "logging_steps": config.get("logging_steps", 10),
            "num_generations": config.get("num_generations", 4),
            "bf16": torch.cuda.is_bf16_supported(),
            "gradient_checkpointing": config.get("gradient_checkpointing", True),
            # GRPO algorithm parameters (TRL built-in)
            "loss_type": config.get("loss_type", "dapo"),
            "beta": config.get("beta", 0.0),
            "epsilon": config.get("epsilon", 0.2),
            "scale_rewards": config.get("scale_rewards", "group"),
            "mask_truncated_completions": config.get("mask_truncated_completions", False),
        }

        # Only add epsilon_high if specified (None means use TRL default behavior)
        epsilon_high = config.get("epsilon_high")
        if epsilon_high is not None:
            grpo_kwargs["epsilon_high"] = epsilon_high

        training_args = GRPOConfig(**grpo_kwargs)

        # Configure LoRA if enabled
        peft_config = None
        if config.get("use_lora", False):
            from peft import LoraConfig, TaskType

            # Default target modules for common architectures
            target_modules = config.get("lora_target_modules")
            if target_modules is None:
                # Auto-detect common attention modules
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

            peft_config = LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.05),
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            print(f"LoRA enabled: r={peft_config.r}, alpha={peft_config.lora_alpha}, targets={target_modules}")

        # Initialize trainer
        trainer_kwargs = {
            "model": model_name,
            "reward_funcs": dummy_reward_func,
            "args": training_args,
            "train_dataset": dataset,
        }
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config

        self.trainer = GRPOTrainer(**trainer_kwargs)

        self.model = self.trainer.model
        self.tokenizer = self.trainer.tokenizer

        # Create optimizer and scheduler explicitly
        # (normally done lazily in trainer.train(), but we use custom train_step)
        num_training_steps = (
            len(dataset) // config.get("batch_size", 8) * config.get("num_epochs", 5)
        )
        if config.get("max_steps", -1) > 0:
            num_training_steps = min(num_training_steps, config.get("max_steps", -1))

        self.trainer.create_optimizer_and_scheduler(
            num_training_steps=num_training_steps
        )

        self.initialized = True

        print("ActorWorker initialized successfully")
        return True

    @modal.method()
    def initialize(self, config: dict, resume_from: Optional[str] = None) -> bool:
        """Initialize GRPOTrainer with config (Modal method wrapper).

        Args:
            config: Configuration dictionary with training parameters
            resume_from: Optional checkpoint path to resume from

        Returns:
            True if initialization successful
        """
        return self._do_initialize(config, resume_from)

    @modal.method()
    def train_full(self) -> dict:
        """Run the full training loop using TRL's built-in trainer.

        This uses TRL's internal training loop. For custom veRL-style training,
        use train_step instead.

        Returns:
            Training metrics
        """
        if not self.initialized:
            raise RuntimeError("ActorWorker not initialized. Call initialize() first.")

        print("Starting full training loop...")
        self.trainer.train()

        # Commit volume changes
        volume.commit()

        return {"status": "completed"}

    @modal.method()
    def train_step(
        self,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
        old_logprobs: Optional[list[list[float]]] = None,
        config: Optional[dict] = None,
    ) -> dict:
        """Single training step with pre-computed generations and rewards.

        This method implements TRL's GRPO loss computation with:
        - Importance sampling (ratio of new/old log probs)
        - Policy clipping based on loss_type (grpo, dapo, bnpo, dr_grpo, cispo, sapo)
        - KL penalty when beta != 0
        - Proper advantage normalization (group/batch/none)
        - Different aggregation strategies per loss_type

        Args:
            prompts: List of prompts
            completions: List of completions (one per prompt)
            rewards: List of rewards for each completion
            old_logprobs: Optional per-token log probabilities from the rollout model
                         Shape: list of lists, each inner list is token log probs for one completion
            config: Optional config dict to initialize if not already initialized

        Returns:
            Dictionary with loss and metrics
        """
        # Auto-initialize if config provided and not yet initialized
        if not self.initialized:
            if config is None:
                raise RuntimeError(
                    "ActorWorker not initialized. Provide config or call initialize() first."
                )
            print("Auto-initializing ActorWorker...")
            self._do_initialize(config)

        import torch

        # Get GRPO parameters from config
        loss_type = self.config.get("loss_type", "dapo")
        beta = self.config.get("beta", 0.0)
        epsilon_low = self.config.get("epsilon", 0.2)
        # epsilon_high defaults to epsilon_low if not specified or None
        epsilon_high = self.config.get("epsilon_high")
        if epsilon_high is None:
            epsilon_high = epsilon_low
        scale_rewards = self.config.get("scale_rewards", "group")
        num_generations = self.config.get("num_generations", 4)

        # Max completion length for training (truncate long completions to save memory)
        # This is separate from max_tokens for generation
        max_completion_length = min(
            self.config.get("max_completion_length", 1024),
            self.config.get("max_tokens", 8000)
        )

        # Clear CUDA cache before processing
        torch.cuda.empty_cache()

        # Tokenize inputs (prompts)
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,  # Limit prompt length for training
        ).to(self.model.device)

        # Tokenize completions (truncate to save memory)
        completion_inputs = self.tokenizer(
            completions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_completion_length,
        ).to(self.model.device)

        # Forward pass to get current policy log probs
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=torch.cat(
                    [inputs.input_ids, completion_inputs.input_ids], dim=1
                ),
                attention_mask=torch.cat(
                    [inputs.attention_mask, completion_inputs.attention_mask], dim=1
                ),
            )

        # Extract logits for completion tokens only
        logits = outputs.logits[:, inputs.input_ids.shape[1] - 1 : -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Get per-token log probs for actual completion tokens
        completion_tokens = completion_inputs.input_ids
        per_token_logps = torch.gather(
            log_probs, dim=-1, index=completion_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Mask for valid (non-padding) tokens
        mask = completion_inputs.attention_mask.float()

        # === Compute Advantages ===
        rewards_tensor = torch.tensor(
            rewards, device=self.model.device, dtype=torch.float32
        )

        # Group-level advantage normalization (GRPO style)
        batch_size = rewards_tensor.shape[0]
        num_groups = batch_size // num_generations if num_generations > 1 else batch_size

        if scale_rewards == "group" and num_generations > 1:
            # Reshape to (num_groups, num_generations) for group statistics
            grouped_rewards = rewards_tensor.view(num_groups, num_generations)
            mean_grouped = grouped_rewards.mean(dim=1, keepdim=True)
            std_grouped = grouped_rewards.std(dim=1, keepdim=True)
            # Normalize within groups
            advantages = (grouped_rewards - mean_grouped) / (std_grouped + 1e-8)
            advantages = advantages.view(-1)  # Flatten back
        elif scale_rewards == "batch":
            # Batch-level normalization
            advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        else:
            # No normalization
            advantages = rewards_tensor

        # Expand advantages to (B, 1) for broadcasting with per-token losses
        advantages = advantages.unsqueeze(-1)

        # === Compute Importance Sampling Ratio ===
        if old_logprobs is not None:
            # Pad old_logprobs to match completion length
            max_len = per_token_logps.shape[1]
            old_per_token_logps = torch.zeros_like(per_token_logps)
            for i, old_lp in enumerate(old_logprobs):
                if old_lp is not None and len(old_lp) > 0:
                    length = min(len(old_lp), max_len)
                    old_per_token_logps[i, :length] = torch.tensor(
                        old_lp[:length], device=self.model.device, dtype=per_token_logps.dtype
                    )

            # Log ratio for importance sampling
            log_ratio = per_token_logps - old_per_token_logps
            coef_1 = torch.exp(log_ratio)  # Importance sampling ratio
        else:
            # No old logprobs - assume on-policy (ratio = 1)
            coef_1 = torch.ones_like(per_token_logps)
            log_ratio = torch.zeros_like(per_token_logps)

        # === Compute Per-Token Loss Based on Loss Type ===
        if loss_type == "cispo":
            # CISPO: Clip importance weights, multiply with log probs
            clamped_ratios = torch.clamp(coef_1, max=1 + epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps

        elif loss_type == "sapo":
            # SAPO: Soft adaptive with temperature control
            sapo_temp_neg = 1.05
            sapo_temp_pos = 1.0

            per_token_loss = torch.empty_like(coef_1)
            positive_mask = (advantages > 0).expand_as(coef_1)

            # Soft clipping with sigmoid
            def sapo_token_loss(ratio, temperature):
                sigmoid_input = temperature * (ratio - 1)
                return torch.sigmoid(sigmoid_input) * 4 / temperature

            per_token_loss[positive_mask] = sapo_token_loss(
                coef_1[positive_mask], sapo_temp_pos
            )
            per_token_loss[~positive_mask] = sapo_token_loss(
                coef_1[~positive_mask], sapo_temp_neg
            )
            per_token_loss = -per_token_loss * advantages

        else:
            # GRPO/DAPO/BNPO/DR_GRPO: Two-sided clipping
            # Clip ratio between (1 - epsilon_low, 1 + epsilon_high)
            coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

            # PPO-style loss: min of clipped and unclipped
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # === Add KL Penalty (if beta != 0) ===
        if beta != 0.0 and old_logprobs is not None:
            # KL divergence: E_p[log(p/q)] approximated as exp(log_q - log_p) - (log_q - log_p) - 1
            per_token_kl = torch.exp(-log_ratio) - (-log_ratio) - 1
            per_token_loss = per_token_loss + beta * per_token_kl

        # === Aggregate Loss Based on Loss Type ===
        if loss_type in ["grpo", "sapo"]:
            # Normalize by sequence length (per-sequence mean, then batch mean)
            seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
            loss = ((per_token_loss * mask).sum(dim=-1) / seq_lengths).mean()

        elif loss_type == "bnpo":
            # Normalize by total token count in batch
            total_tokens = mask.sum().clamp(min=1.0)
            loss = (per_token_loss * mask).sum() / total_tokens

        elif loss_type == "dr_grpo":
            # Normalize by (batch_size * max_completion_length)
            max_completion_length = self.config.get("max_tokens", 512)
            loss = (per_token_loss * mask).sum() / (batch_size * max_completion_length)

        elif loss_type in ["cispo", "dapo"]:
            # Normalize by total tokens across batch (global normalization)
            total_tokens = mask.sum().clamp(min=1.0)
            loss = (per_token_loss * mask).sum() / total_tokens

        else:
            # Fallback: simple mean
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)

        # === Backward Pass ===
        self.trainer.accelerator.backward(loss)
        self.trainer.optimizer.step()
        self.trainer.optimizer.zero_grad()

        # === Compute Metrics ===
        with torch.no_grad():
            # Clip ratio for logging
            mean_ratio = coef_1[mask.bool()].mean().item() if mask.sum() > 0 else 1.0
            clip_frac = ((coef_1 < 1 - epsilon_low) | (coef_1 > 1 + epsilon_high)).float()
            clip_frac = (clip_frac * mask).sum() / mask.sum().clamp(min=1.0)

            # KL divergence
            if old_logprobs is not None:
                approx_kl = (log_ratio * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                approx_kl = torch.tensor(0.0)

        return {
            "loss": loss.item(),
            "mean_reward": rewards_tensor.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "mean_ratio": mean_ratio,
            "clip_fraction": clip_frac.item(),
            "approx_kl": approx_kl.item(),
            "loss_type": loss_type,
        }

    @modal.method()
    def get_weights(self, config: Optional[dict] = None) -> bytes:
        """Return model state dict (serialized) for sync to rollout workers.

        NOTE: This method transfers large amounts of data over the network.
        For optimal performance, use sync_weights_to_volume() instead.

        Returns:
            Serialized model state dict
        """
        if not self.initialized:
            if config is None:
                raise RuntimeError(
                    "ActorWorker not initialized. Provide config or call initialize() first."
                )
            print("Auto-initializing ActorWorker for get_weights...")
            self._do_initialize(config)

        import torch

        buffer = io.BytesIO()
        # Get the underlying model (unwrap from any wrappers)
        model_to_save = self.trainer.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        torch.save(model_to_save.state_dict(), buffer)
        return buffer.getvalue()

    @modal.method()
    def sync_weights_to_volume(self, sync_id: int, config: Optional[dict] = None) -> dict:
        """Save model weights to volume for rollout workers to load.

        This is the optimal weight sync method for Modal - it avoids transferring
        large state dicts over the network by writing directly to the shared volume.

        Args:
            sync_id: Unique identifier for this sync (e.g., training step)
            config: Optional config dict to initialize if not already initialized

        Returns:
            Dict with sync metadata (path, sync_id, model_name)
        """
        if not self.initialized:
            if config is None:
                raise RuntimeError(
                    "ActorWorker not initialized. Provide config or call initialize() first."
                )
            print("Auto-initializing ActorWorker for sync_weights_to_volume...")
            self._do_initialize(config)

        import json
        import os
        import time

        from safetensors.torch import save_file

        # Fixed path for weight sync (not checkpoints - those are separate)
        sync_dir = f"{STORAGE_PATH}/weight_sync"
        os.makedirs(sync_dir, exist_ok=True)

        # Get the underlying model
        model_to_save = self.trainer.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        # Save weights in safetensors format (faster loading)
        weights_path = f"{sync_dir}/model.safetensors"
        print(f"Saving weights to {weights_path}...")
        start_time = time.time()

        # Handle LoRA models - merge weights for vLLM compatibility
        if self.config.get("use_lora", False):
            from peft import PeftModel
            if isinstance(model_to_save, PeftModel):
                print("  Merging LoRA weights for vLLM compatibility...")
                # Merge LoRA adapters into base weights
                model_to_save.merge_adapter()

                # Get state dict and filter/rename keys for vLLM compatibility
                raw_state_dict = model_to_save.state_dict()
                state_dict = {}

                for key, value in raw_state_dict.items():
                    # Skip LoRA-specific parameters (lora_A, lora_B weights)
                    if "lora_" in key:
                        continue

                    # Remove base_model.model. prefix
                    clean_key = key.replace("base_model.model.", "")

                    # Remove .base_layer from key names (LoRA wraps original layers)
                    clean_key = clean_key.replace(".base_layer", "")

                    state_dict[clean_key] = value

                # Unmerge to allow continued LoRA training
                model_to_save.unmerge_adapter()
            else:
                state_dict = model_to_save.state_dict()
                state_dict = {
                    k.replace("base_model.model.", "").replace("base_model.", ""): v
                    for k, v in state_dict.items()
                }
        else:
            state_dict = model_to_save.state_dict()

        # Convert to CPU and ensure contiguous for safetensors
        state_dict_cpu = {k: v.cpu().contiguous() for k, v in state_dict.items()}
        save_file(state_dict_cpu, weights_path)

        save_time = time.time() - start_time
        print(f"Weights saved in {save_time:.2f}s")

        # Write sync manifest with metadata
        manifest = {
            "sync_id": sync_id,
            "timestamp": time.time(),
            "model_name": self.config.get("model_name", "unknown"),
            "weights_path": weights_path,
        }
        manifest_path = f"{sync_dir}/manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Commit volume to make weights available to other workers
        volume.commit()

        return manifest

    @modal.method()
    def sync_weights_to_model_path(
        self, model_path: str, sync_id: int, config: Optional[dict] = None
    ) -> dict:
        """Save model weights to a specific model path for vLLM reload_weights.

        This is the optimal weight sync for vLLM v1's reload_weights pattern:
        1. Actor saves weights to the model path vLLM was initialized with
        2. Rollout worker calls reload_weights() to reload from that path

        Args:
            model_path: Path where vLLM model is loaded from
            sync_id: Unique identifier for this sync (e.g., training step)
            config: Optional config dict to initialize if not already initialized

        Returns:
            Dict with sync metadata
        """
        if not self.initialized:
            if config is None:
                raise RuntimeError(
                    "ActorWorker not initialized. Provide config or call initialize() first."
                )
            print("Auto-initializing ActorWorker for sync_weights_to_model_path...")
            self._do_initialize(config)

        import json
        import os
        import time

        from safetensors.torch import save_file

        print(f"Saving weights to model path: {model_path}")
        start_time = time.time()

        # Ensure directory exists
        os.makedirs(model_path, exist_ok=True)

        # Get the underlying model
        model_to_save = self.trainer.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        # Handle LoRA models - merge weights for vLLM compatibility
        weights_path = f"{model_path}/model.safetensors"
        if self.config.get("use_lora", False):
            from peft import PeftModel
            if isinstance(model_to_save, PeftModel):
                print("  Merging LoRA weights for vLLM compatibility...")
                # Merge LoRA adapters into base weights
                model_to_save.merge_adapter()

                # Get state dict and filter/rename keys for vLLM compatibility
                raw_state_dict = model_to_save.state_dict()
                state_dict = {}

                for key, value in raw_state_dict.items():
                    # Skip LoRA-specific parameters (lora_A, lora_B weights)
                    if "lora_" in key:
                        continue

                    # Remove base_model.model. prefix
                    clean_key = key.replace("base_model.model.", "")

                    # Remove .base_layer from key names (LoRA wraps original layers)
                    clean_key = clean_key.replace(".base_layer", "")

                    state_dict[clean_key] = value

                # Unmerge to allow continued LoRA training
                model_to_save.unmerge_adapter()
            else:
                state_dict = model_to_save.state_dict()
                state_dict = {
                    k.replace("base_model.model.", "").replace("base_model.", ""): v
                    for k, v in state_dict.items()
                }
        else:
            state_dict = model_to_save.state_dict()

        state_dict_cpu = {k: v.cpu().contiguous() for k, v in state_dict.items()}
        save_file(state_dict_cpu, weights_path)

        save_time = time.time() - start_time
        print(f"Weights saved to {weights_path} in {save_time:.2f}s")

        # Write sync manifest
        manifest = {
            "sync_id": sync_id,
            "timestamp": time.time(),
            "model_name": self.config.get("model_name", "unknown"),
            "weights_path": weights_path,
            "model_path": model_path,
        }
        manifest_path = f"{model_path}/sync_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Commit volume
        volume.commit()

        return manifest

    @modal.method()
    def get_weights_chunked(
        self,
        chunk_size_mb: int = 100,
        config: Optional[dict] = None,
    ) -> list[tuple[str, bytes]]:
        """Return model weights as serialized chunks for streaming transfer.

        For large models, this allows memory-efficient weight transfer by
        yielding parameters in smaller chunks.

        Args:
            chunk_size_mb: Target chunk size in MB
            config: Optional config dict to initialize if not already initialized

        Returns:
            List of (param_name, param_bytes) tuples per chunk
        """
        if not self.initialized:
            if config is None:
                raise RuntimeError(
                    "ActorWorker not initialized. Provide config or call initialize() first."
                )
            print("Auto-initializing ActorWorker for get_weights_chunked...")
            self._do_initialize(config)

        import torch

        # Get the underlying model
        model_to_save = self.trainer.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        chunks = []
        current_chunk = []
        current_size = 0

        for name, param in model_to_save.named_parameters():
            # Serialize parameter
            buffer = io.BytesIO()
            torch.save(param.data.cpu(), buffer)
            param_bytes = buffer.getvalue()

            current_chunk.append((name, param_bytes))
            current_size += len(param_bytes)

            # Yield chunk if it exceeds target size
            if current_size >= chunk_size_bytes:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        print(f"Split weights into {len(chunks)} chunks")
        return chunks

    @modal.method()
    def save_checkpoint(self, step: int, config: Optional[dict] = None) -> str:
        """Save checkpoint to volume.

        Args:
            step: Current training step
            config: Optional config dict to initialize if not already initialized

        Returns:
            Path to saved checkpoint
        """
        if not self.initialized:
            if config is None:
                raise RuntimeError(
                    "ActorWorker not initialized. Provide config or call initialize() first."
                )
            print("Auto-initializing ActorWorker for save_checkpoint...")
            self._do_initialize(config)

        checkpoint_path = f"{STORAGE_PATH}/checkpoints/step-{step}"
        self.trainer.save_model(checkpoint_path)
        volume.commit()

        print(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    @modal.method()
    def get_current_step(self) -> int:
        """Get current training step."""
        if not self.initialized:
            return 0
        return self.trainer.state.global_step
