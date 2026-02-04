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

        # Training arguments
        training_args = GRPOConfig(
            output_dir=config.get("checkpoint_dir", f"{STORAGE_PATH}/checkpoints"),
            use_vllm=False,  # We handle vLLM separately via RolloutWorkers
            report_to="none",  # Disable wandb in actor (orchestrator handles it)
            per_device_train_batch_size=config.get("batch_size", 8),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            learning_rate=config.get("learning_rate", 5e-6),
            num_train_epochs=config.get("num_epochs", 5),
            max_steps=config.get("max_steps", -1),
            save_steps=config.get("save_steps", 100),
            logging_steps=config.get("logging_steps", 10),
            num_generations=config.get("num_generations", 4),
            bf16=torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
        )

        # Initialize trainer
        self.trainer = GRPOTrainer(
            model=model_name,
            reward_funcs=dummy_reward_func,
            args=training_args,
            train_dataset=dataset,
        )

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

        This method allows for veRL-style external orchestration where
        generation and reward computation happen on separate workers.

        Args:
            prompts: List of prompts
            completions: List of completions (one per prompt)
            rewards: List of rewards for each completion
            old_logprobs: Optional log probabilities from the rollout model
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

        # This is a simplified version - full implementation would need to
        # integrate more deeply with GRPOTrainer's internal methods
        # For now, we use the trainer's built-in step

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("max_model_len", 4096),
        ).to(self.model.device)

        # Tokenize completions
        completion_inputs = self.tokenizer(
            completions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("max_tokens", 512),
        ).to(self.model.device)

        # Forward pass
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=torch.cat(
                    [inputs.input_ids, completion_inputs.input_ids], dim=1
                ),
                attention_mask=torch.cat(
                    [inputs.attention_mask, completion_inputs.attention_mask], dim=1
                ),
            )

        # Compute loss (simplified GRPO loss)
        logits = outputs.logits[:, inputs.input_ids.shape[1] - 1 : -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Get log probs for actual tokens
        completion_tokens = completion_inputs.input_ids
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=completion_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding
        mask = completion_inputs.attention_mask
        token_log_probs = token_log_probs * mask

        # Sum log probs per sequence
        seq_log_probs = token_log_probs.sum(dim=-1)

        # Convert rewards to tensor
        rewards_tensor = torch.tensor(
            rewards, device=self.model.device, dtype=torch.float32
        )

        # GRPO loss: -E[r * log_prob]
        loss = -(rewards_tensor * seq_log_probs).mean()

        # Backward pass
        self.trainer.accelerator.backward(loss)
        self.trainer.optimizer.step()
        self.trainer.optimizer.zero_grad()

        return {
            "loss": loss.item(),
            "mean_reward": rewards_tensor.mean().item(),
            "mean_log_prob": seq_log_probs.mean().item(),
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

        # Save weights in safetensors format
        weights_path = f"{model_path}/model.safetensors"
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
