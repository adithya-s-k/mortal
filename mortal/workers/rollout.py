"""Rollout worker for vLLM inference."""

from typing import Optional

import modal

# Import app and resources from the shared app module
from mortal.app import app, volume, VLLM_IMAGE

STORAGE_PATH = "/storage"


@app.cls(
    image=VLLM_IMAGE,
    gpu="A10G",
    volumes={STORAGE_PATH: volume},
    scaledown_window=300,  # Keep warm for 5 minutes
    timeout=600,  # 10 minute timeout per method call
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
)
class RolloutWorker:
    """Standalone vLLM inference worker for generating completions.

    This worker uses vLLM for fast inference and can be scaled horizontally.
    It maintains a cached model and can reload from checkpoints when the
    actor's weights are updated.
    """

    @modal.enter()
    def setup(self):
        """Initialize on container start."""
        import os

        # Use v1 engine for sleep/wake_up/reload_weights support
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Set multiprocessing to spawn mode
        import multiprocessing

        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set

        self.llm = None
        self.current_model_path = None
        self.tokenizer = None
        # Cache for config/tokenizer to avoid repeated downloads
        self._cached_base_model = None
        self._cached_config_path = None
        # Track current sync ID to avoid unnecessary reloads
        self._current_sync_id = None
        # Local model path for efficient reload_weights
        self._local_model_path = None
        self._container_id = id(self)
        print(f"RolloutWorker container started (lazy vLLM init, v1 engine enabled), "
              f"container_id={self._container_id}")

    def _load_model(
        self,
        model_path: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
    ):
        """Load or reload vLLM engine.

        Args:
            model_path: Path to model (HuggingFace name or local checkpoint)
            max_model_len: Maximum sequence length
            gpu_memory_utilization: Fraction of GPU memory to use
        """
        from vllm import LLM
        import torch

        if model_path == self.current_model_path and self.llm is not None:
            return  # Already loaded

        # Clean up existing model
        if self.llm is not None:
            print(f"Unloading current model: {self.current_model_path}")
            del self.llm
            torch.cuda.empty_cache()

        print(f"Loading model: {model_path}")
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,  # Disable CUDA graphs for stability
        )
        self.current_model_path = model_path
        print(f"Model loaded successfully: {model_path}")

    @modal.method()
    def generate(
        self,
        prompts: list[str],
        model_path: str = "Qwen/Qwen2-0.5B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
        max_model_len: int = 4096,
    ) -> dict:
        """Generate completions with logprobs.

        Args:
            prompts: List of prompts to generate from
            model_path: Model to use (HF name or checkpoint path)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            n: Number of completions per prompt
            max_model_len: Maximum model context length

        Returns:
            Dictionary with completions and logprobs
        """
        from vllm import SamplingParams

        print(f"[RolloutWorker.generate] id(self)={id(self)}, "
              f"_local_model_path={self._local_model_path}, "
              f"model_path={model_path}, n_prompts={len(prompts)}")
        self._load_model(model_path, max_model_len=max_model_len)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            logprobs=1,  # Return top-1 logprob for each token
        )

        print(f"Generating {len(prompts)} prompts with n={n}...")
        outputs = self.llm.generate(prompts, sampling_params)

        # Process outputs
        completions = []
        all_logprobs = []
        prompt_indices = []  # Track which prompt each completion belongs to

        for prompt_idx, output in enumerate(outputs):
            for completion in output.outputs:
                completions.append(completion.text)
                prompt_indices.append(prompt_idx)

                # Extract logprobs
                if completion.logprobs:
                    token_logprobs = []
                    for lp in completion.logprobs:
                        if lp:
                            # Get the logprob of the sampled token
                            token_logprobs.append(list(lp.values())[0].logprob)
                    all_logprobs.append(token_logprobs)
                else:
                    all_logprobs.append([])

        print(f"Generated {len(completions)} completions")
        return {
            "completions": completions,
            "logprobs": all_logprobs,
            "prompt_indices": prompt_indices,
        }

    @modal.method()
    def reload_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Reload model from volume checkpoint.

        Args:
            checkpoint_path: Path to checkpoint on the volume

        Returns:
            True if reload successful
        """
        import torch

        print(f"Reloading from checkpoint: {checkpoint_path}")

        # Free GPU memory first
        if self.llm is not None:
            del self.llm
            self.llm = None
            self.current_model_path = None
            torch.cuda.empty_cache()

        # Get latest files from volume
        volume.reload()

        # Load model from checkpoint
        self._load_model(checkpoint_path)
        return True

    @modal.method()
    def initialize_for_weight_sync(
        self,
        base_model: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
    ) -> str:
        """Initialize model from local volume path for efficient weight sync.

        This prepares the model for efficient weight updates by:
        1. Copying base model to a local volume path
        2. Initializing vLLM with that local path
        3. Enabling reload_weights() to reload from the same path

        Call update_weights_from_volume() to update weights after this.

        Args:
            base_model: HuggingFace model name to initialize from
            max_model_len: Maximum model context length
            gpu_memory_utilization: GPU memory fraction to use

        Returns:
            Local model path that weights should be saved to
        """
        import os
        import shutil
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from huggingface_hub import snapshot_download
        from vllm import LLM

        # Local path for this model (on volume)
        model_name_safe = base_model.replace("/", "_")
        local_path = f"{STORAGE_PATH}/model_cache/{model_name_safe}"

        print(f"Initializing model at local path: {local_path}")

        # Check if we already have this model cached
        need_download = False
        if not os.path.exists(local_path):
            need_download = True
        else:
            # Validate cached model - check for corrupted weights from LoRA runs
            safetensors_path = f"{local_path}/model.safetensors"
            if os.path.exists(safetensors_path):
                try:
                    from safetensors import safe_open
                    with safe_open(safetensors_path, framework="pt") as f:
                        keys = list(f.keys())[:5]
                        if any("base_model" in k or "lora_" in k for k in keys):
                            print(f"  ⚠ Cached model has corrupted weights (LoRA artifacts)")
                            print(f"  Clearing cache and re-downloading...")
                            import shutil
                            shutil.rmtree(local_path)
                            need_download = True
                except Exception as e:
                    print(f"  Warning: Could not validate cache ({e}), using as-is")
                    volume.reload()
            else:
                volume.reload()

        if need_download:
            print(f"  Downloading {base_model} to volume...")
            os.makedirs(local_path, exist_ok=True)

            # Download model files
            downloaded_path = snapshot_download(
                base_model,
                local_dir=local_path,
                ignore_patterns=["*.md", "*.txt", ".git*"],
            )
            print(f"  Model cached at {local_path}")
            volume.commit()
        elif os.path.exists(local_path):
            print(f"  Using cached model at {local_path}")

        # Free GPU memory if model exists
        if self.llm is not None:
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()

        # Initialize vLLM with local path
        print(f"  Loading vLLM from {local_path}...")
        self.llm = LLM(
            model=local_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
            trust_remote_code=True,
        )

        self.current_model_path = local_path
        self._local_model_path = local_path
        self._cached_base_model = base_model

        print(f"Model initialized for weight sync at {local_path}, id(self)={id(self)}")
        return local_path

    @modal.method()
    def update_weights_from_volume(self, weights_path: str = None) -> bool:
        """Update weights using vLLM's reload_weights (efficient, no model recreation).

        This uses vLLM v1's sleep/wake_up/reload_weights pattern:
        1. sleep(level=2) - frees GPU memory
        2. wake_up(tags=["weights"]) - allocates weights memory only
        3. reload_weights() - reloads from the local model path
        4. wake_up(tags=["kv_cache"]) - reallocates KV cache

        The weights_path should contain updated model.safetensors file.

        Args:
            weights_path: Path to updated weights (default: use local model path)

        Returns:
            True if update successful
        """
        import os
        import time

        print(f"[RolloutWorker.update_weights_from_volume] called with weights_path={weights_path}")
        print(f"  self.llm={self.llm is not None}, self._local_model_path={self._local_model_path}")
        print(f"  self.current_model_path={self.current_model_path}, id(self)={id(self)}")

        if self.llm is None or self._local_model_path is None:
            print(f"Error: Model not initialized. Call initialize_for_weight_sync first.")
            print(f"  DEBUG: This likely means Modal routed this call to a NEW container replica.")
            print(f"  DEBUG: The original container (with initialized model) may be busy with generate().")
            return False

        start_time = time.time()
        model_path = weights_path or self._local_model_path

        print(f"Updating weights via reload_weights (no model recreation)...")

        try:
            if self._is_vllm_v1():
                # v1 engine: use sleep/wake_up for efficient memory management
                print("  Using vLLM v1 sleep/wake_up pattern...")

                # Sleep FIRST to release file handles, then reload volume
                self.llm.sleep(level=2)

                # Now volume.reload() won't fail with "open files"
                volume.reload()

                # Check weights file exists
                weights_file = f"{model_path}/model.safetensors"
                if not os.path.exists(weights_file):
                    print(f"Error: Weights file not found at {weights_file}")
                    self.llm.wake_up()
                    return False

                # Reallocate weights memory only (avoid OOM)
                self.llm.wake_up(tags=["weights"])

                # Reload weights from local path (no args = reload from original path)
                self.llm.collective_rpc("reload_weights")

                # Reallocate KV cache
                self.llm.wake_up(tags=["kv_cache"])

                elapsed = time.time() - start_time
                print(f"  Weights updated in {elapsed:.2f}s (efficient reload)")
                return True
            else:
                # v0 engine: need to recreate model
                print("  vLLM v0 detected, using full model reload...")
                import torch
                del self.llm
                self.llm = None
                torch.cuda.empty_cache()
                volume.reload()
                self._load_model(model_path)
                elapsed = time.time() - start_time
                print(f"  Weights updated in {elapsed:.2f}s (full reload)")
                return True

        except Exception as e:
            print(f"Error updating weights: {e}")
            import traceback
            traceback.print_exc()
            return False

    @modal.method()
    def load_from_weight_sync(
        self,
        base_model: str,
        sync_dir: str = "/storage/weight_sync",
        max_model_len: int = 4096,
    ) -> bool:
        """Load model with weights from the weight sync directory.

        This method is optimized for the volume-based sync approach:
        - First time: Downloads config/tokenizer from base model, caches them
        - Subsequent times: Uses cached config, only loads updated weights

        Args:
            base_model: Base HuggingFace model name (for config/tokenizer)
            sync_dir: Directory containing synced weights
            max_model_len: Maximum model length

        Returns:
            True if load successful
        """
        import json
        import os
        import shutil

        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM

        # Reload volume to get latest weights
        volume.reload()

        # Check if weights exist
        weights_path = f"{sync_dir}/model.safetensors"
        manifest_path = f"{sync_dir}/manifest.json"

        if not os.path.exists(weights_path):
            print(f"Warning: No weights found at {weights_path}")
            return False

        # Read manifest
        sync_id = None
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
                sync_id = manifest.get("sync_id")

        # Check if we need to reload (same sync_id means no change)
        if sync_id is not None and sync_id == self._current_sync_id:
            print(f"Weights already at sync_id {sync_id}, skipping reload")
            return True

        print(f"Loading weights from {weights_path} (sync_id: {sync_id})")

        # Ensure we have config/tokenizer cached
        config_cache_dir = f"{sync_dir}/config_cache"
        if self._cached_base_model != base_model or not os.path.exists(config_cache_dir):
            print(f"Caching config/tokenizer from {base_model}...")
            os.makedirs(config_cache_dir, exist_ok=True)

            config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
            config.save_pretrained(config_cache_dir)

            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            tokenizer.save_pretrained(config_cache_dir)

            self._cached_base_model = base_model
            self._cached_config_path = config_cache_dir

        # Copy weights to config cache dir for vLLM to load
        shutil.copy(weights_path, f"{config_cache_dir}/model.safetensors")

        # Free GPU memory
        if self.llm is not None:
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()

        # Load model from the combined directory
        self.llm = LLM(
            model=config_cache_dir,
            gpu_memory_utilization=0.90,
            max_model_len=max_model_len,
            enforce_eager=True,
            trust_remote_code=True,
        )

        self.current_model_path = config_cache_dir
        self._current_sync_id = sync_id
        print(f"Model loaded from weight sync (sync_id: {sync_id})")
        return True

    @modal.method()
    def update_weights(self, weights_bytes: bytes) -> bool:
        """Update model weights from serialized state dict (legacy disk-based method).

        NOTE: This method writes to disk. For better performance, use
        update_weights_direct() which updates vLLM weights in-place.

        Args:
            weights_bytes: Serialized state dict bytes

        Returns:
            True if update successful
        """
        import io
        import os
        import torch
        from safetensors.torch import save_file

        if self.llm is None:
            print("Warning: No model loaded, cannot update weights")
            return False

        print("Updating model weights from actor (disk-based)...")

        # First, free GPU memory by deleting vLLM engine
        original_model_path = self.current_model_path
        del self.llm
        self.llm = None
        torch.cuda.empty_cache()

        # Load state dict to CPU to avoid OOM
        buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=True)

        # vLLM doesn't support direct weight updates, so we need to
        # save to disk and reload. This is a limitation of vLLM.
        temp_path = f"{STORAGE_PATH}/temp_weights"
        os.makedirs(temp_path, exist_ok=True)

        # Save weights in safetensors format (preferred by vLLM)
        save_file(state_dict, f"{temp_path}/model.safetensors")

        # Copy config files from the original model if it's a HF model
        # This is needed for vLLM to properly load the model
        from transformers import AutoConfig, AutoTokenizer

        try:
            # Load and save config/tokenizer from original model
            config = AutoConfig.from_pretrained(original_model_path)
            config.save_pretrained(temp_path)

            tokenizer = AutoTokenizer.from_pretrained(original_model_path)
            tokenizer.save_pretrained(temp_path)
        except Exception as e:
            print(f"Warning: Could not copy config/tokenizer: {e}")
            print("Falling back to original model path for reload")
            # Clean up and reload original model
            del state_dict
            torch.cuda.empty_cache()
            self._load_model(original_model_path)
            return False

        # Clean up state dict from CPU memory
        del state_dict
        torch.cuda.empty_cache()

        # Reload vLLM with updated weights
        self._load_model(temp_path)
        print("Weights updated successfully")
        return True

    def _is_vllm_v1(self) -> bool:
        """Check if vLLM v1 engine is being used.

        v1 has collective_rpc and sleep/wake_up methods.

        Returns:
            True if v1 engine, False otherwise
        """
        if self.llm is None:
            return False
        return hasattr(self.llm, "collective_rpc") and hasattr(self.llm, "sleep")

    def _do_disk_based_update(self, weights_bytes: bytes) -> bool:
        """Internal disk-based weight update (called when direct update fails).

        Args:
            weights_bytes: Serialized state dict bytes

        Returns:
            True if update successful
        """
        import io
        import os

        import torch
        from safetensors.torch import save_file

        print("Using disk-based weight update (fallback)...")

        # First, free GPU memory by deleting vLLM engine
        original_model_path = self.current_model_path
        del self.llm
        self.llm = None
        torch.cuda.empty_cache()

        # Load state dict to CPU
        buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=True)

        # Save to disk
        temp_path = f"{STORAGE_PATH}/temp_weights"
        os.makedirs(temp_path, exist_ok=True)
        save_file(state_dict, f"{temp_path}/model.safetensors")

        # Copy config files from the original model
        from transformers import AutoConfig, AutoTokenizer

        try:
            config = AutoConfig.from_pretrained(original_model_path)
            config.save_pretrained(temp_path)

            tokenizer = AutoTokenizer.from_pretrained(original_model_path)
            tokenizer.save_pretrained(temp_path)
        except Exception as e:
            print(f"Warning: Could not copy config/tokenizer: {e}")
            del state_dict
            torch.cuda.empty_cache()
            self._load_model(original_model_path)
            return False

        del state_dict
        torch.cuda.empty_cache()

        self._load_model(temp_path)
        print("Weights updated via disk-based fallback")
        return True

    def _create_weights_iterator(self, state_dict: dict):
        """Create a weights iterator from state dict for vLLM reload_weights.

        Args:
            state_dict: Model state dictionary

        Yields:
            Tuples of (name, tensor) for each parameter
        """
        import torch

        for name, param in state_dict.items():
            # Fix parameter name for vLLM compatibility
            fixed_name = name
            for prefix in ["_checkpoint_wrapped_module.", "module.", "model."]:
                if fixed_name.startswith(prefix):
                    fixed_name = fixed_name[len(prefix):]

            # Ensure tensor is on correct device and contiguous
            if param.device.type == "cpu":
                param = param.to("cuda")
            yield (fixed_name, param.contiguous())

    @modal.method()
    def update_weights_direct(self, weights_bytes: bytes) -> bool:
        """Update model weights directly in vLLM memory (no disk I/O).

        This is the optimal weight sync method. It uses:
        - vLLM v1: collective_rpc("reload_weights") with sleep/wake_up for memory management
        - vLLM v0: Direct model.load_weights() (deprecated but fallback)

        Args:
            weights_bytes: Serialized state dict bytes (torch.save format)

        Returns:
            True if update successful
        """
        import io
        import time

        import torch

        if self.llm is None:
            print("Warning: No model loaded, cannot update weights")
            return False

        print("Updating model weights directly in vLLM...")
        start_time = time.time()

        # Load state dict to CPU first
        buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=True)

        try:
            if self._is_vllm_v1():
                # vLLM v1: Use collective_rpc with sleep/wake_up pattern
                print("  Using vLLM v1 API (collective_rpc + sleep/wake_up)")

                # Create weights iterator
                def weights_iter():
                    return self._create_weights_iterator(state_dict)

                # Sleep to free GPU memory (level=2: discard weights and KV cache)
                self.llm.sleep(level=2)

                # Reallocate weights memory only
                self.llm.wake_up(tags=["weights"])

                # Load weights in-place via collective_rpc
                self.llm.collective_rpc(
                    "reload_weights",
                    kwargs={"weights_iterator": weights_iter()}
                )

                # Reallocate KV cache
                self.llm.wake_up(tags=["kv_cache"])

                elapsed = time.time() - start_time
                print(f"  Weights updated via v1 API in {elapsed:.2f}s (no disk I/O)")

            else:
                # vLLM v0 (deprecated): Try direct model access
                print("  Using vLLM v0 API (direct model access)")

                # Try different API paths for v0
                llm_model = None
                paths_to_try = [
                    lambda: self.llm.llm_engine.model_executor.driver_worker.model_runner.model,
                    lambda: self.llm.llm_engine.model_executor.model,
                    lambda: self.llm.llm_engine.driver_worker.model_runner.model,
                ]

                for path_fn in paths_to_try:
                    try:
                        llm_model = path_fn()
                        if llm_model is not None and hasattr(llm_model, 'load_weights'):
                            break
                    except (AttributeError, TypeError):
                        continue

                if llm_model is None:
                    print("  Warning: Could not access vLLM internal model")
                    raise RuntimeError("Cannot access vLLM model for weight update")

                # Create weights list for load_weights
                weights_to_load = list(self._create_weights_iterator(state_dict))

                # Load weights directly
                llm_model.load_weights(weights_to_load)

                # Reset KV cache
                try:
                    self.llm.reset_prefix_cache()
                except (AttributeError, TypeError):
                    pass

                elapsed = time.time() - start_time
                print(f"  Weights updated via v0 API in {elapsed:.2f}s (no disk I/O)")

            # Clean up
            del state_dict
            torch.cuda.empty_cache()
            return True

        except Exception as e:
            print(f"Warning: Direct weight update failed: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to disk-based update")
            del state_dict
            torch.cuda.empty_cache()
            return self._do_disk_based_update(weights_bytes)

    @modal.method()
    def update_weights_streamed(
        self,
        weight_chunk: list[tuple[str, bytes]],
        is_last_chunk: bool = False,
    ) -> bool:
        """Update model weights in streaming fashion (memory efficient).

        For large models, this allows transferring weights in smaller chunks
        to avoid memory pressure from holding the entire state dict.

        Args:
            weight_chunk: List of (param_name, param_bytes) tuples
            is_last_chunk: If True, reset KV cache after this chunk

        Returns:
            True if update successful
        """
        import io

        import torch

        if self.llm is None:
            print("Warning: No model loaded, cannot update weights")
            return False

        # Get vLLM's internal model
        try:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        except AttributeError as e:
            print(f"Warning: Could not access vLLM internal model: {e}")
            return False

        # Process this chunk
        weights_to_load = []
        for name, param_bytes in weight_chunk:
            # Deserialize parameter
            buffer = io.BytesIO(param_bytes)
            param = torch.load(buffer, map_location="cuda", weights_only=True)

            # Fix parameter name
            fixed_name = name
            for prefix in ["_checkpoint_wrapped_module.", "module.", "model."]:
                if fixed_name.startswith(prefix):
                    fixed_name = fixed_name[len(prefix):]

            weights_to_load.append((fixed_name, param.contiguous()))

        # Load this chunk of weights
        try:
            llm_model.load_weights(weights_to_load)
        except Exception as e:
            print(f"Warning: Failed to load weight chunk: {e}")
            return False

        # Clean up
        del weights_to_load
        torch.cuda.empty_cache()

        # Reset cache after last chunk
        if is_last_chunk:
            try:
                self.llm.reset_prefix_cache()
            except AttributeError:
                pass
            print("Streamed weight update complete")

        return True

    @modal.method()
    def health_check(self) -> dict:
        """Check worker health and return status.

        Returns:
            Dictionary with health status
        """
        return {
            "status": "healthy",
            "model_loaded": self.llm is not None,
            "current_model": self.current_model_path,
        }

    @modal.method()
    def compute_logprobs(
        self,
        prompts: list[str],
        completions: list[str],
        model_path: str = "Qwen/Qwen2-0.5B-Instruct",
        max_model_len: int = 4096,
    ) -> list[list[float]]:
        """Compute log probabilities for given prompt-completion pairs.

        This is useful for computing reference model log probs or
        importance weights.

        Args:
            prompts: List of prompts
            completions: List of completions (one per prompt)
            model_path: Model to use
            max_model_len: Maximum context length

        Returns:
            List of log probability lists (one per completion)
        """
        from vllm import SamplingParams

        self._load_model(model_path, max_model_len=max_model_len)

        # Combine prompts and completions
        full_texts = [p + c for p, c in zip(prompts, completions)]

        # Use prompt_logprobs to get logprobs for the completion tokens
        sampling_params = SamplingParams(
            max_tokens=1,  # We don't need to generate, just compute logprobs
            temperature=1.0,
            prompt_logprobs=1,
        )

        outputs = self.llm.generate(full_texts, sampling_params)

        all_logprobs = []
        for i, output in enumerate(outputs):
            if output.prompt_logprobs:
                # Get logprobs for completion portion only
                prompt_len = len(self.llm.get_tokenizer().encode(prompts[i]))
                completion_logprobs = []
                for j, lp in enumerate(output.prompt_logprobs):
                    if j >= prompt_len and lp:
                        completion_logprobs.append(list(lp.values())[0].logprob)
                all_logprobs.append(completion_logprobs)
            else:
                all_logprobs.append([])

        return all_logprobs
