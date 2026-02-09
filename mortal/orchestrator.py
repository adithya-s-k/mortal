"""Main training orchestrator - veRL style coordination of workers."""

from typing import Optional

import modal

from mortal.app import app, volume, TRAINING_IMAGE
from mortal.config import OrchestratorConfig, AsyncConfig
from mortal.workers.actor import ActorWorker
from mortal.workers.rollout import RolloutWorker
from mortal.rewards import compute_rewards
from mortal.rewards.base import RewardEnvironment

STORAGE_PATH = "/storage"


def get_batch(dataset, batch_idx: int, batch_size: int) -> dict:
    """Get a batch from the dataset.

    Args:
        dataset: HuggingFace dataset
        batch_idx: Batch index
        batch_size: Batch size

    Returns:
        Dictionary with prompts and testcases
    """
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(dataset))

    batch = dataset.select(range(start_idx, end_idx))
    return {
        "prompts": batch["prompt"],
        "testcases": batch["testcases"],
    }


def chunk_list(lst: list, num_chunks: int) -> list[list]:
    """Split a list into roughly equal chunks.

    Args:
        lst: List to split
        num_chunks: Number of chunks

    Returns:
        List of chunks
    """
    if num_chunks <= 0:
        return [lst]

    chunk_size = max(1, len(lst) // num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        if i == num_chunks - 1:
            # Last chunk gets any remainder
            chunks.append(lst[start:])
        else:
            chunks.append(lst[start : start + chunk_size])

    return [c for c in chunks if c]  # Remove empty chunks


@app.function(
    image=TRAINING_IMAGE,
    volumes={STORAGE_PATH: volume},
    timeout=3600 * 24,  # 24 hours
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
)
def train(config_dict: Optional[dict] = None, reward_funcs=None, train_dataset=None):
    """Main training orchestrator - veRL style.

    Coordinates the training loop:
    1. Get batch of prompts from dataset
    2. Rollout workers generate completions (parallel)
    3. Reward workers score completions (parallel via Sandboxes or custom)
    4. Actor computes GRPO loss and updates
    5. Sync weights to rollout workers periodically
    6. Checkpoint periodically

    Args:
        config_dict: Configuration dictionary (optional, uses defaults if None)
        reward_funcs: Reward function(s) - None/"sandbox" for Modal Sandbox,
            callable or list of callables for custom rewards.
        train_dataset: Optional HuggingFace Dataset with "prompt" column.
            If provided, used directly (no column renaming).
            If None, loaded from config with default column renaming.

    Returns:
        Training result summary
    """
    import wandb
    from datasets import load_dataset

    # Parse config
    if config_dict is None:
        config_dict = {}
    config = OrchestratorConfig.from_dict(config_dict)

    print(f"Starting training with config: {config.to_dict()}")

    # Initialize wandb
    wandb.init(
        project="modal-grpo-trl",
        config=config.to_dict(),
        name=f"grpo-{config.model.model_name.split('/')[-1]}",
    )

    # Load dataset
    if train_dataset is not None:
        if callable(train_dataset):
            print("Running dataset prep function on container...")
            dataset = train_dataset()
        else:
            dataset = train_dataset
        if config.max_samples:
            dataset = dataset.select(range(min(config.max_samples, len(dataset))))
        print(f"Using provided dataset with {len(dataset)} samples")
    else:
        print("Loading dataset...")
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.dataset_split,
        )
        dataset = dataset.rename_column("instruction", "prompt")
        dataset = dataset.rename_column("testcase", "testcases")

        if config.max_samples:
            dataset = dataset.select(range(min(config.max_samples, len(dataset))))

        print(f"Dataset loaded with {len(dataset)} samples")

    # Initialize workers with configurable GPU types
    actor_gpu = config.actor_gpu
    rollout_gpu = config.rollout_gpu
    print(f"Initializing workers (actor_gpu={actor_gpu}, rollout_gpu={rollout_gpu})...")
    actor = ActorWorker.with_options(gpu=actor_gpu)()

    # Initialize actor
    actor.initialize.remote(config.to_dict())

    # Create rollout workers
    rollout_workers = [
        RolloutWorker.with_options(gpu=rollout_gpu)()
        for _ in range(config.num_rollout_workers)
    ]

    # Pre-warm rollout workers with appropriate initialization
    sync_method = config.training.weight_sync_method
    print(f"Pre-warming rollout workers (sync_method: {sync_method})...")

    # Track model path for reload-based sync
    rollout_model_path = None

    if sync_method == "reload":
        # Initialize with local model path for efficient reload_weights
        print("  Using efficient reload-based initialization...")
        warmup_futures = []
        for worker in rollout_workers:
            future = worker.initialize_for_weight_sync.spawn(
                base_model=config.model.model_name,
                max_model_len=config.model.max_model_len,
            )
            warmup_futures.append(future)

        # Wait for initialization and get the model path
        for future in warmup_futures:
            rollout_model_path = future.get()  # All workers return same path
        print(f"  Rollout workers initialized at {rollout_model_path}")

        # Do a quick warmup generation
        warmup_gen_futures = []
        for worker in rollout_workers:
            future = worker.generate.spawn(
                prompts=["Hello"],
                model_path=rollout_model_path,
                max_tokens=10,
                max_model_len=config.model.max_model_len,
            )
            warmup_gen_futures.append(future)
        for future in warmup_gen_futures:
            future.get()
    else:
        # Standard warmup with HuggingFace model path
        warmup_futures = []
        for worker in rollout_workers:
            future = worker.generate.spawn(
                prompts=["Hello"],
                model_path=config.model.model_name,
                max_tokens=10,
                max_model_len=config.model.max_model_len,
            )
            warmup_futures.append(future)

        # Wait for warmup
        for future in warmup_futures:
            future.get()

    print("Rollout workers warmed up")

    # Calculate training steps
    batch_size = config.training.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    total_steps = num_batches * config.training.num_epochs

    if config.training.max_steps > 0:
        total_steps = min(total_steps, config.training.max_steps)

    print(f"Total training steps: {total_steps}")

    # Track current model path for rollout workers (updated after checkpoint syncs)
    # For "reload" mode, use the local volume path; otherwise use HuggingFace name
    if sync_method == "reload" and rollout_model_path is not None:
        current_rollout_model_path = rollout_model_path
    else:
        current_rollout_model_path = config.model.model_name

    # If prompts are chat-format (list of dicts), apply chat template to convert to strings
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, trust_remote_code=True)

    sample_prompt = dataset[0]["prompt"]
    if isinstance(sample_prompt, list) and len(sample_prompt) > 0 and isinstance(sample_prompt[0], dict):
        print("Detected chat-format prompts, applying chat template...")
        def apply_chat_template(example):
            example["prompt"] = tokenizer.apply_chat_template(
                example["prompt"], tokenize=False, add_generation_prompt=True,
            )
            return example
        dataset = dataset.map(apply_chat_template)

    # Determine extra columns for reward kwargs
    extra_columns = [c for c in dataset.column_names if c != "prompt"]

    # Wrap reward_funcs if custom (RewardEnvironment instances → TRL-compatible callables)
    if reward_funcs is not None and reward_funcs != "sandbox":
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        def _wrap_env(env):
            """Wrap a RewardEnvironment into a callable for compute_rewards."""
            def _reward(completions, **kw):
                # Extract text from TRL chat format if needed
                texts = []
                for c in completions:
                    if isinstance(c, list) and len(c) > 0 and isinstance(c[0], dict):
                        texts.append(c[0]["content"])
                    else:
                        texts.append(c)
                prompts = kw.pop("prompts", [""] * len(texts))
                print(f"[_wrap_env] Calling {getattr(env, 'name', 'RewardEnv')}.score_batch "
                      f"with {len(texts)} completions")
                try:
                    result = env.score_batch(texts, prompts, **kw)
                    print(f"[_wrap_env] Result: {result}")
                    return result
                except Exception as e:
                    print(f"[_wrap_env] ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    return [0.0] * len(texts)
            _reward.__name__ = getattr(env, "name", "RewardEnv")
            return _reward

        reward_funcs = [
            _wrap_env(rf) if isinstance(rf, RewardEnvironment) else rf
            for rf in reward_funcs
        ]

    # Training loop
    global_step = 0
    for epoch in range(config.training.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.training.num_epochs} ===")

        for batch_idx in range(num_batches):
            if (
                config.training.max_steps > 0
                and global_step >= config.training.max_steps
            ):
                break

            # 1. Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = dataset.select(range(start_idx, end_idx))
            prompts = batch["prompt"]
            batch_kwargs = {col: batch[col] for col in extra_columns}

            print(
                f"\nStep {global_step + 1}/{total_steps}: Processing {len(prompts)} prompts"
            )

            # 2. Generate completions (parallel across rollout workers)
            # Expand prompts for multiple generations per prompt
            expanded_prompts = []
            expanded_kwargs = {col: [] for col in extra_columns}
            for i, p in enumerate(prompts):
                for _ in range(config.training.num_generations):
                    expanded_prompts.append(p)
                    for col in extra_columns:
                        expanded_kwargs[col].append(batch_kwargs[col][i])

            # Chunk prompts across workers
            prompt_chunks = chunk_list(expanded_prompts, config.num_rollout_workers)

            generation_futures = []
            for i, worker in enumerate(rollout_workers):
                if i < len(prompt_chunks) and prompt_chunks[i]:
                    future = worker.generate.spawn(
                        prompts=prompt_chunks[i],
                        model_path=current_rollout_model_path,
                        max_tokens=config.generation.max_tokens,
                        temperature=config.generation.temperature,
                        top_p=config.generation.top_p,
                        n=1,  # Already expanded prompts
                        max_model_len=config.model.max_model_len,
                    )
                    generation_futures.append(future)

            # Collect generation results
            all_completions = []
            all_logprobs = []
            for future in generation_futures:
                result = future.get()
                all_completions.extend(result["completions"])
                all_logprobs.extend(result["logprobs"])

            print(f"Generated {len(all_completions)} completions")

            # 3. Compute rewards (parallel via sandboxes or custom reward_funcs)
            # Wrap plain string completions into TRL chat format so reward
            # functions written for TRL work identically in distributed mode
            wrapped_completions = [[{"role": "assistant", "content": c}] for c in all_completions]
            print("Computing rewards...")
            rewards = compute_rewards(
                reward_funcs=reward_funcs,
                completions=wrapped_completions,
                prompts=expanded_prompts,
                **expanded_kwargs,
            )
            mean_reward = sum(rewards) / len(rewards) if rewards else 0

            print(f"Mean reward: {mean_reward:.4f}")

            # 4. Train step
            print("Performing training step...")
            loss_result = actor.train_step.remote(
                prompts=expanded_prompts,
                completions=all_completions,
                rewards=rewards,
                old_logprobs=all_logprobs,
                config=config.to_dict(),  # Pass config for auto-init if needed
            )

            # Log metrics
            metrics = {
                "train/loss": loss_result.get("loss", 0),
                "train/mean_reward": mean_reward,
                "train/mean_advantage": loss_result.get("mean_advantage", 0),
                "train/mean_ratio": loss_result.get("mean_ratio", 1.0),
                "train/clip_fraction": loss_result.get("clip_fraction", 0),
                "train/approx_kl": loss_result.get("approx_kl", 0),
                "train/epoch": epoch,
                "train/step": global_step,
            }
            wandb.log(metrics, step=global_step)

            # Print key metrics
            print(f"  Loss: {loss_result.get('loss', 0):.4f}, "
                  f"KL: {loss_result.get('approx_kl', 0):.4f}, "
                  f"Clip: {loss_result.get('clip_fraction', 0):.2%}")

            # 5. Sync weights to rollout workers (periodically)
            if (global_step + 1) % config.training.sync_weights_every == 0:
                sync_method = config.training.weight_sync_method
                print(f"Syncing weights to rollout workers (method: {sync_method})...")

                try:
                    import time
                    sync_start = time.time()

                    if sync_method == "direct":
                        # DIRECT: In-memory weight update via vLLM's load_weights()
                        # Fastest method - no disk I/O, updates weights in-place
                        weights_bytes = actor.get_weights.remote(config=config.to_dict())
                        get_weights_time = time.time() - sync_start
                        weights_size_mb = len(weights_bytes) / (1024 * 1024)
                        print(f"  Got weights: {weights_size_mb:.1f} MB in {get_weights_time:.2f}s")

                        # Update all rollout workers directly (parallel)
                        sync_futures = []
                        for worker in rollout_workers:
                            future = worker.update_weights_direct.spawn(weights_bytes)
                            sync_futures.append(future)

                        # Wait for all workers to complete
                        sync_results = [f.get() for f in sync_futures]
                        success_count = sum(sync_results)

                        sync_time = time.time() - sync_start
                        print(f"  Weights synced to {success_count}/{len(rollout_workers)} workers in {sync_time:.2f}s")

                    elif sync_method == "volume":
                        # VOLUME: Save to shared volume, workers load from volume
                        # Good balance of speed and reliability, avoids large network transfers
                        manifest = actor.sync_weights_to_volume.remote(
                            sync_id=global_step + 1,
                            config=config.to_dict(),
                        )
                        print(f"  Weights saved to volume (sync_id: {manifest['sync_id']})")

                        # Reload rollout workers from volume using optimized method
                        sync_futures = []
                        for worker in rollout_workers:
                            future = worker.load_from_weight_sync.spawn(
                                base_model=config.model.model_name,
                                sync_dir="/storage/weight_sync",
                                max_model_len=config.model.max_model_len,
                            )
                            sync_futures.append(future)

                        sync_results = [f.get() for f in sync_futures]
                        success_count = sum(sync_results)

                        sync_time = time.time() - sync_start
                        print(f"  Weights synced via volume in {sync_time:.2f}s")

                    elif sync_method == "reload":
                        # RELOAD: Save weights to model path, use vLLM reload_weights
                        # Most efficient - uses sleep/wake_up/reload_weights pattern
                        if rollout_model_path is None:
                            print("  Warning: rollout_model_path not set, falling back to volume sync")
                            sync_method = "volume"
                        else:
                            # Save weights to the model path rollout workers are using
                            manifest = actor.sync_weights_to_model_path.remote(
                                model_path=rollout_model_path,
                                sync_id=global_step + 1,
                                config=config.to_dict(),
                            )
                            print(f"  Weights saved to {rollout_model_path}")

                            # Reload weights in rollout workers (uses sleep/wake_up pattern)
                            sync_futures = []
                            for worker in rollout_workers:
                                future = worker.update_weights_from_volume.spawn(
                                    weights_path=rollout_model_path,
                                )
                                sync_futures.append(future)

                            sync_results = [f.get() for f in sync_futures]
                            success_count = sum(sync_results)

                            sync_time = time.time() - sync_start
                            print(f"  Weights reloaded in {sync_time:.2f}s (efficient reload)")

                    elif sync_method == "checkpoint":
                        # CHECKPOINT: Full checkpoint save + reload (most reliable, slowest)
                        sync_checkpoint_path = actor.save_checkpoint.remote(
                            step=global_step + 1,
                            config=config.to_dict(),
                        )
                        print(f"  Checkpoint saved to: {sync_checkpoint_path}")

                        # Reload rollout workers from checkpoint
                        sync_futures = []
                        for worker in rollout_workers:
                            future = worker.reload_from_checkpoint.spawn(sync_checkpoint_path)
                            sync_futures.append(future)

                        sync_results = [f.get() for f in sync_futures]
                        success_count = sum(sync_results)

                        # Update model path for future generate calls
                        current_rollout_model_path = sync_checkpoint_path

                        sync_time = time.time() - sync_start
                        print(f"  Weights synced via checkpoint in {sync_time:.2f}s")

                    else:
                        print(f"  Warning: Unknown sync method '{sync_method}', skipping")
                        success_count = 0

                    if success_count < len(rollout_workers):
                        print(f"  Warning: {len(rollout_workers) - success_count} workers failed to sync")

                except Exception as e:
                    print(f"Warning: Weight sync failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Continuing with current rollout weights...")

            # 6. Checkpoint periodically
            if (global_step + 1) % config.training.save_steps == 0:
                print("Saving checkpoint...")
                checkpoint_path = actor.save_checkpoint.remote(global_step + 1, config=config.to_dict())
                print(f"Checkpoint saved: {checkpoint_path}")

            global_step += 1

    # Final checkpoint
    print("\nSaving final checkpoint...")
    final_checkpoint = actor.save_checkpoint.remote(global_step, config=config.to_dict())

    # Commit volume
    volume.commit()

    wandb.finish()

    return {
        "status": "completed",
        "total_steps": global_step,
        "final_checkpoint": final_checkpoint,
    }


@app.cls(
    image=TRAINING_IMAGE,
    gpu="A100",
    volumes={STORAGE_PATH: volume},
    timeout=3600 * 24,
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
)
class SingleNodeTrainer:
    """Single-node trainer using TRL's GRPOTrainer directly."""

    @modal.method()
    def run(self, config_dict: Optional[dict] = None, reward_funcs=None, train_dataset=None):
        """Single-node training with optional vLLM support.

        Args:
            config_dict: Configuration dictionary.
            reward_funcs: Optional reward function(s) — callable or list of callables.
                If provided, passed directly to GRPOTrainer (TRL-compatible).
                If None, falls back to sandbox/local reward based on mode.reward_type.
            train_dataset: Optional HuggingFace Dataset with "prompt" column.
                If provided, used directly (no column renaming).
                If None, loaded from config with default column renaming.
        """
        import os
        import subprocess

        from mortal.config import OrchestratorConfig, SingleNode as SingleNodeMode

        if config_dict is None:
            config_dict = {}
        config = OrchestratorConfig.from_dict(config_dict)
        mode = config.mode
        assert isinstance(mode, SingleNodeMode), f"SingleNodeTrainer requires SingleNode mode, got {type(mode)}"

        print(f"Starting single-node training: model={config.model.model_name}, "
              f"gpu={mode.gpu.to_modal_spec()}, use_vllm={mode.use_vllm}, "
              f"vllm_mode={mode.vllm_mode}, reward_type={mode.reward_type}")

        # vLLM setup — MUST happen before importing torch to set CUDA_VISIBLE_DEVICES
        use_vllm = mode.use_vllm
        vllm_mode = None

        if use_vllm:
            if mode.vllm_mode == "serve":
                vllm_mode = "server"
                # vLLM server on GPU 0
                env_copy = os.environ.copy()
                env_copy["CUDA_VISIBLE_DEVICES"] = "0"
                subprocess.Popen(
                    ["trl", "vllm-serve", "--model", config.model.model_name],
                    env=env_copy,
                )
                # Training on remaining GPUs — set BEFORE torch import
                training_gpus = ",".join(str(i) for i in range(1, mode.gpu.count))
                os.environ["CUDA_VISIBLE_DEVICES"] = training_gpus
            elif mode.vllm_mode == "colocate":
                vllm_mode = "colocate"
                os.environ["RANK"] = "0"
                os.environ["LOCAL_RANK"] = "0"
                os.environ["WORLD_SIZE"] = "1"
                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = "12355"

        # Now safe to import torch (CUDA_VISIBLE_DEVICES is already set)
        import torch
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer

        # Load dataset
        if train_dataset is not None:
            if callable(train_dataset):
                print("Running dataset prep function on container...")
                dataset = train_dataset()
            else:
                dataset = train_dataset
            if config.max_samples:
                dataset = dataset.select(range(min(config.max_samples, len(dataset))))
            print(f"Using provided dataset with {len(dataset)} samples")
        else:
            dataset = load_dataset(
                config.dataset_name, config.dataset_config, split=config.dataset_split,
            )
            dataset = dataset.rename_column("instruction", "prompt")
            dataset = dataset.rename_column("testcase", "testcases")
            if config.max_samples:
                dataset = dataset.select(range(min(config.max_samples, len(dataset))))
            print(f"Dataset loaded with {len(dataset)} samples")

        # Build GRPOConfig
        grpo_kwargs = {
            "output_dir": f"{STORAGE_PATH}/checkpoints",
            "report_to": config.training.report_to,
            "use_vllm": use_vllm,
            "per_device_train_batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "learning_rate": config.training.learning_rate,
            "num_train_epochs": config.training.num_epochs,
            "max_steps": config.training.max_steps,
            "save_steps": config.training.save_steps,
            "logging_steps": config.training.logging_steps,
            "num_generations": config.training.num_generations,
            "bf16": torch.cuda.is_bf16_supported(),
            "gradient_checkpointing": config.training.gradient_checkpointing,
            "loss_type": config.training.loss_type,
            "beta": config.training.beta,
            "epsilon": config.training.epsilon,
            "scale_rewards": config.training.scale_rewards,
            "mask_truncated_completions": config.training.mask_truncated_completions,
            "max_completion_length": config.training.max_completion_length,
        }
        if vllm_mode is not None:
            grpo_kwargs["vllm_mode"] = vllm_mode
        if config.training.epsilon_high is not None:
            grpo_kwargs["epsilon_high"] = config.training.epsilon_high

        training_args = GRPOConfig(**grpo_kwargs)

        # LoRA
        peft_config = None
        if config.training.use_lora:
            from peft import LoraConfig, TaskType

            target_modules = config.training.lora_target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ]
            peft_config = LoraConfig(
                r=config.training.lora_r,
                lora_alpha=config.training.lora_alpha,
                lora_dropout=config.training.lora_dropout,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            print(f"LoRA enabled: r={peft_config.r}, alpha={peft_config.lora_alpha}")

        # Select reward function(s)
        if reward_funcs is not None:
            from mortal.rewards.base import RewardEnvironment as _RE

            def _wrap_env(env):
                """Wrap a RewardEnvironment into a TRL-compatible callable."""
                def _trl_reward(completions, **kw):
                    # TRL passes completions as list of list of dicts
                    texts = [c[0]["content"] for c in completions]
                    prompts = kw.pop("prompts", [""] * len(texts))
                    print(f"[_wrap_env] Calling {getattr(env, 'name', 'RewardEnv')}.score_batch "
                          f"with {len(texts)} completions, kwargs keys: {list(kw.keys())}")
                    try:
                        result = env.score_batch(texts, prompts, **kw)
                        print(f"[_wrap_env] Result: {result}")
                        return result
                    except Exception as e:
                        print(f"[_wrap_env] ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        return [0.0] * len(texts)
                _trl_reward.__name__ = getattr(env, "name", "RewardEnv")
                return _trl_reward

            # Normalize to list, wrap any RewardEnvironment instances
            if not isinstance(reward_funcs, list):
                reward_funcs = [reward_funcs]
            selected_reward_fn = [
                _wrap_env(rf) if isinstance(rf, _RE) else rf
                for rf in reward_funcs
            ]
            print(f"Using custom reward function(s): {selected_reward_fn}")
        elif mode.reward_type == "function":
            from mortal.workers.reward import function_reward_function
            print("Using function-based reward execution (pre-warmed Modal Functions)")
            selected_reward_fn = function_reward_function
        elif mode.reward_type == "local":
            from mortal.workers.reward import local_reward_function
            print("Using local (in-process) reward execution")
            selected_reward_fn = local_reward_function
        else:
            from mortal.workers.reward import reward_helper_function
            print("Using sandbox-based reward execution")
            selected_reward_fn = reward_helper_function

        # Create trainer
        trainer_kwargs = {
            "model": config.model.model_name,
            "reward_funcs": selected_reward_fn,
            "args": training_args,
            "train_dataset": dataset,
        }
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config

        trainer = GRPOTrainer(**trainer_kwargs)
        trainer.train()

        volume.commit()

        return {"status": "completed", "mode": "single_node"}


def train_local(config_dict: dict, reward_funcs, train_dataset=None, reward_weights=None):
    """Local orchestrator - runs on user's CPU with custom reward functions.

    Same training loop as train() but:
    - Runs locally (not as a Modal function)
    - Dataset loaded locally or from train_dataset arg
    - Rewards computed locally via compute_rewards()
    - GPU workers (actor, rollout) still run on Modal via .remote()

    Args:
        config_dict: Configuration dictionary
        reward_funcs: Custom reward function(s) - callable, RewardEnvironment,
            or list of either.
        train_dataset: Optional HuggingFace Dataset with "prompt" column.
            If None, loads from config dataset settings.
        reward_weights: Optional list of weights for combining multiple
            reward functions.
    """
    import time

    try:
        import wandb

        has_wandb = True
    except ImportError:
        has_wandb = False

    config = OrchestratorConfig.from_dict(config_dict)

    print(f"Starting local training with config: {config.to_dict()}")

    # Initialize wandb if available
    if has_wandb:
        wandb.init(
            project="modal-grpo-trl",
            config=config.to_dict(),
            name=f"grpo-local-{config.model.model_name.split('/')[-1]}",
        )

    # Load dataset
    if train_dataset is not None:
        if callable(train_dataset):
            print("Running dataset prep function...")
            dataset = train_dataset()
        else:
            dataset = train_dataset
        # Ensure required columns exist
        if "prompt" not in dataset.column_names:
            raise ValueError(
                "train_dataset must have a 'prompt' column. "
                f"Found columns: {dataset.column_names}"
            )
    else:
        from datasets import load_dataset

        print("Loading dataset...")
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.dataset_split,
        )
        dataset = dataset.rename_column("instruction", "prompt")
        dataset = dataset.rename_column("testcase", "testcases")

    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    print(f"Dataset loaded with {len(dataset)} samples")

    # Determine which dataset columns to pass as kwargs to reward functions
    # All columns except "prompt" are passed as keyword arguments
    extra_columns = [c for c in dataset.column_names if c != "prompt"]

    # Initialize workers with configurable GPU types
    actor_gpu = config.actor_gpu
    rollout_gpu = config.rollout_gpu
    print(f"Initializing workers (actor_gpu={actor_gpu}, rollout_gpu={rollout_gpu})...")
    actor = ActorWorker.with_options(gpu=actor_gpu)()
    actor.initialize.remote(config.to_dict())

    rollout_workers = [
        RolloutWorker.with_options(gpu=rollout_gpu)()
        for _ in range(config.num_rollout_workers)
    ]

    # Pre-warm rollout workers
    sync_method = config.training.weight_sync_method
    print(f"Pre-warming rollout workers (sync_method: {sync_method})...")

    rollout_model_path = None

    if sync_method == "reload":
        print("  Using efficient reload-based initialization...")
        warmup_futures = []
        for worker in rollout_workers:
            future = worker.initialize_for_weight_sync.spawn(
                base_model=config.model.model_name,
                max_model_len=config.model.max_model_len,
            )
            warmup_futures.append(future)
        for future in warmup_futures:
            rollout_model_path = future.get()
        print(f"  Rollout workers initialized at {rollout_model_path}")

        warmup_gen_futures = []
        for worker in rollout_workers:
            future = worker.generate.spawn(
                prompts=["Hello"],
                model_path=rollout_model_path,
                max_tokens=10,
                max_model_len=config.model.max_model_len,
            )
            warmup_gen_futures.append(future)
        for future in warmup_gen_futures:
            future.get()
    else:
        warmup_futures = []
        for worker in rollout_workers:
            future = worker.generate.spawn(
                prompts=["Hello"],
                model_path=config.model.model_name,
                max_tokens=10,
                max_model_len=config.model.max_model_len,
            )
            warmup_futures.append(future)
        for future in warmup_futures:
            future.get()

    print("Rollout workers warmed up")

    # Calculate training steps
    batch_size = config.training.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    total_steps = num_batches * config.training.num_epochs

    if config.training.max_steps > 0:
        total_steps = min(total_steps, config.training.max_steps)

    print(f"Total training steps: {total_steps}")

    if sync_method == "reload" and rollout_model_path is not None:
        current_rollout_model_path = rollout_model_path
    else:
        current_rollout_model_path = config.model.model_name

    # Call setup() on RewardEnvironment instances
    reward_envs = []
    if isinstance(reward_funcs, list):
        reward_envs = [rf for rf in reward_funcs if isinstance(rf, RewardEnvironment)]
    elif isinstance(reward_funcs, RewardEnvironment):
        reward_envs = [reward_funcs]
    for env in reward_envs:
        env.setup()

    # Training loop
    global_step = 0
    for epoch in range(config.training.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.training.num_epochs} ===")

        for batch_idx in range(num_batches):
            if (
                config.training.max_steps > 0
                and global_step >= config.training.max_steps
            ):
                break

            # 1. Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = dataset.select(range(start_idx, end_idx))
            prompts = batch["prompt"]

            # Gather extra columns for reward kwargs
            batch_kwargs = {col: batch[col] for col in extra_columns}

            print(
                f"\nStep {global_step + 1}/{total_steps}: Processing {len(prompts)} prompts"
            )

            # 2. Generate completions (parallel across rollout workers)
            expanded_prompts = []
            expanded_kwargs = {col: [] for col in extra_columns}
            for i, p in enumerate(prompts):
                for _ in range(config.training.num_generations):
                    expanded_prompts.append(p)
                    for col in extra_columns:
                        expanded_kwargs[col].append(batch_kwargs[col][i])

            prompt_chunks = chunk_list(expanded_prompts, config.num_rollout_workers)

            generation_futures = []
            for i, worker in enumerate(rollout_workers):
                if i < len(prompt_chunks) and prompt_chunks[i]:
                    future = worker.generate.spawn(
                        prompts=prompt_chunks[i],
                        model_path=current_rollout_model_path,
                        max_tokens=config.generation.max_tokens,
                        temperature=config.generation.temperature,
                        top_p=config.generation.top_p,
                        n=1,
                        max_model_len=config.model.max_model_len,
                    )
                    generation_futures.append(future)

            all_completions = []
            all_logprobs = []
            for future in generation_futures:
                result = future.get()
                all_completions.extend(result["completions"])
                all_logprobs.extend(result["logprobs"])

            print(f"Generated {len(all_completions)} completions")

            # 3. Compute rewards locally via custom reward_funcs
            print("Computing rewards (local)...")
            rewards = compute_rewards(
                reward_funcs=reward_funcs,
                completions=all_completions,
                prompts=expanded_prompts,
                weights=reward_weights,
                **expanded_kwargs,
            )
            mean_reward = sum(rewards) / len(rewards) if rewards else 0

            print(f"Mean reward: {mean_reward:.4f}")

            # 4. Train step (on Modal GPU)
            print("Performing training step...")
            loss_result = actor.train_step.remote(
                prompts=expanded_prompts,
                completions=all_completions,
                rewards=rewards,
                old_logprobs=all_logprobs,
                config=config.to_dict(),
            )

            # Log metrics
            metrics = {
                "train/loss": loss_result.get("loss", 0),
                "train/mean_reward": mean_reward,
                "train/mean_advantage": loss_result.get("mean_advantage", 0),
                "train/mean_ratio": loss_result.get("mean_ratio", 1.0),
                "train/clip_fraction": loss_result.get("clip_fraction", 0),
                "train/approx_kl": loss_result.get("approx_kl", 0),
                "train/epoch": epoch,
                "train/step": global_step,
            }
            if has_wandb:
                wandb.log(metrics, step=global_step)

            print(
                f"  Loss: {loss_result.get('loss', 0):.4f}, "
                f"KL: {loss_result.get('approx_kl', 0):.4f}, "
                f"Clip: {loss_result.get('clip_fraction', 0):.2%}"
            )

            # 5. Sync weights to rollout workers (periodically)
            if (global_step + 1) % config.training.sync_weights_every == 0:
                sync_method = config.training.weight_sync_method
                print(f"Syncing weights to rollout workers (method: {sync_method})...")

                try:
                    sync_start = time.time()

                    if sync_method == "direct":
                        weights_bytes = actor.get_weights.remote(config=config.to_dict())
                        sync_futures = []
                        for worker in rollout_workers:
                            future = worker.update_weights_direct.spawn(weights_bytes)
                            sync_futures.append(future)
                        sync_results = [f.get() for f in sync_futures]
                        success_count = sum(sync_results)

                    elif sync_method == "volume":
                        manifest = actor.sync_weights_to_volume.remote(
                            sync_id=global_step + 1,
                            config=config.to_dict(),
                        )
                        sync_futures = []
                        for worker in rollout_workers:
                            future = worker.load_from_weight_sync.spawn(
                                base_model=config.model.model_name,
                                sync_dir="/storage/weight_sync",
                                max_model_len=config.model.max_model_len,
                            )
                            sync_futures.append(future)
                        sync_results = [f.get() for f in sync_futures]
                        success_count = sum(sync_results)

                    elif sync_method == "reload":
                        if rollout_model_path is None:
                            print("  Warning: rollout_model_path not set, skipping sync")
                            success_count = 0
                        else:
                            manifest = actor.sync_weights_to_model_path.remote(
                                model_path=rollout_model_path,
                                sync_id=global_step + 1,
                                config=config.to_dict(),
                            )
                            sync_futures = []
                            for worker in rollout_workers:
                                future = worker.update_weights_from_volume.spawn(
                                    weights_path=rollout_model_path,
                                )
                                sync_futures.append(future)
                            sync_results = [f.get() for f in sync_futures]
                            success_count = sum(sync_results)

                    elif sync_method == "checkpoint":
                        sync_checkpoint_path = actor.save_checkpoint.remote(
                            step=global_step + 1,
                            config=config.to_dict(),
                        )
                        sync_futures = []
                        for worker in rollout_workers:
                            future = worker.reload_from_checkpoint.spawn(sync_checkpoint_path)
                            sync_futures.append(future)
                        sync_results = [f.get() for f in sync_futures]
                        success_count = sum(sync_results)
                        current_rollout_model_path = sync_checkpoint_path

                    else:
                        print(f"  Warning: Unknown sync method '{sync_method}', skipping")
                        success_count = 0

                    sync_time = time.time() - sync_start
                    print(f"  Weights synced in {sync_time:.2f}s ({success_count}/{len(rollout_workers)} workers)")

                except Exception as e:
                    print(f"Warning: Weight sync failed: {e}")
                    import traceback
                    traceback.print_exc()

            # 6. Checkpoint periodically
            if (global_step + 1) % config.training.save_steps == 0:
                print("Saving checkpoint...")
                checkpoint_path = actor.save_checkpoint.remote(global_step + 1, config=config.to_dict())
                print(f"Checkpoint saved: {checkpoint_path}")

            global_step += 1

    # Call teardown() on RewardEnvironment instances
    for env in reward_envs:
        env.teardown()

    # Final checkpoint
    print("\nSaving final checkpoint...")
    final_checkpoint = actor.save_checkpoint.remote(global_step, config=config.to_dict())

    # volume.commit() is only valid inside a Modal container, skip when local
    try:
        volume.commit()
    except RuntimeError:
        pass

    if has_wandb:
        wandb.finish()

    return {
        "status": "completed",
        "total_steps": global_step,
        "final_checkpoint": final_checkpoint,
    }


# ---------------------------------------------------------------------------
# Shared helpers for async modes
# ---------------------------------------------------------------------------


def _setup_distributed(config_dict, reward_funcs, train_dataset):
    """Common setup for distributed training (sync, pipeline, queue).

    Returns a dict with all initialized components:
        config, dataset, extra_columns, actor, rollout_workers,
        current_rollout_model_path, rollout_model_path, reward_funcs,
        total_steps, batch_size, num_batches
    """
    import wandb
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if config_dict is None:
        config_dict = {}
    config = OrchestratorConfig.from_dict(config_dict)

    print(f"Starting training with config: {config.to_dict()}")

    wandb.init(
        project="modal-grpo-trl",
        config=config.to_dict(),
        name=f"grpo-{config.model.model_name.split('/')[-1]}",
    )

    # Load dataset
    if train_dataset is not None:
        if callable(train_dataset):
            print("Running dataset prep function on container...")
            dataset = train_dataset()
        else:
            dataset = train_dataset
        if config.max_samples:
            dataset = dataset.select(range(min(config.max_samples, len(dataset))))
        print(f"Using provided dataset with {len(dataset)} samples")
    else:
        print("Loading dataset...")
        dataset = load_dataset(
            config.dataset_name, config.dataset_config, split=config.dataset_split,
        )
        dataset = dataset.rename_column("instruction", "prompt")
        dataset = dataset.rename_column("testcase", "testcases")
        if config.max_samples:
            dataset = dataset.select(range(min(config.max_samples, len(dataset))))
        print(f"Dataset loaded with {len(dataset)} samples")

    # Initialize workers
    actor_gpu = config.actor_gpu
    rollout_gpu = config.rollout_gpu
    print(f"Initializing workers (actor_gpu={actor_gpu}, rollout_gpu={rollout_gpu})...")
    actor = ActorWorker.with_options(gpu=actor_gpu)()
    actor.initialize.remote(config.to_dict())

    rollout_workers = [
        RolloutWorker.with_options(gpu=rollout_gpu)()
        for _ in range(config.num_rollout_workers)
    ]

    # Pre-warm rollout workers
    sync_method = config.training.weight_sync_method
    print(f"Pre-warming rollout workers (sync_method: {sync_method})...")
    rollout_model_path = None

    if sync_method == "reload":
        print("  Using efficient reload-based initialization...")
        warmup_futures = []
        for worker in rollout_workers:
            future = worker.initialize_for_weight_sync.spawn(
                base_model=config.model.model_name,
                max_model_len=config.model.max_model_len,
            )
            warmup_futures.append(future)
        for future in warmup_futures:
            rollout_model_path = future.get()
        print(f"  Rollout workers initialized at {rollout_model_path}")

        warmup_gen_futures = []
        for worker in rollout_workers:
            future = worker.generate.spawn(
                prompts=["Hello"], model_path=rollout_model_path,
                max_tokens=10, max_model_len=config.model.max_model_len,
            )
            warmup_gen_futures.append(future)
        for future in warmup_gen_futures:
            future.get()
    else:
        warmup_futures = []
        for worker in rollout_workers:
            future = worker.generate.spawn(
                prompts=["Hello"], model_path=config.model.model_name,
                max_tokens=10, max_model_len=config.model.max_model_len,
            )
            warmup_futures.append(future)
        for future in warmup_futures:
            future.get()

    print("Rollout workers warmed up")

    # Chat template conversion
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, trust_remote_code=True)
    sample_prompt = dataset[0]["prompt"]
    if isinstance(sample_prompt, list) and len(sample_prompt) > 0 and isinstance(sample_prompt[0], dict):
        print("Detected chat-format prompts, applying chat template...")
        def apply_chat_template(example):
            example["prompt"] = tokenizer.apply_chat_template(
                example["prompt"], tokenize=False, add_generation_prompt=True,
            )
            return example
        dataset = dataset.map(apply_chat_template)

    extra_columns = [c for c in dataset.column_names if c != "prompt"]

    # Wrap reward_funcs
    if reward_funcs is not None and reward_funcs != "sandbox":
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        def _wrap_env(env):
            def _reward(completions, **kw):
                texts = []
                for c in completions:
                    if isinstance(c, list) and len(c) > 0 and isinstance(c[0], dict):
                        texts.append(c[0]["content"])
                    else:
                        texts.append(c)
                prompts = kw.pop("prompts", [""] * len(texts))
                try:
                    return env.score_batch(texts, prompts, **kw)
                except Exception as e:
                    print(f"[_wrap_env] ERROR in {getattr(env, 'name', 'RewardEnv')}: {e}")
                    import traceback
                    traceback.print_exc()
                    return [0.0] * len(texts)
            _reward.__name__ = getattr(env, "name", "RewardEnv")
            return _reward

        reward_funcs = [
            _wrap_env(rf) if isinstance(rf, RewardEnvironment) else rf
            for rf in reward_funcs
        ]

    # Calculate steps
    batch_size = config.training.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    total_steps = num_batches * config.training.num_epochs
    if config.training.max_steps > 0:
        total_steps = min(total_steps, config.training.max_steps)

    if sync_method == "reload" and rollout_model_path is not None:
        current_rollout_model_path = rollout_model_path
    else:
        current_rollout_model_path = config.model.model_name

    print(f"Total training steps: {total_steps}")

    return {
        "config": config,
        "dataset": dataset,
        "extra_columns": extra_columns,
        "actor": actor,
        "rollout_workers": rollout_workers,
        "current_rollout_model_path": current_rollout_model_path,
        "rollout_model_path": rollout_model_path,
        "reward_funcs": reward_funcs,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "num_batches": num_batches,
    }


def _get_batch_data(dataset, batch_idx, batch_size, extra_columns, num_generations):
    """Get a batch and expand prompts for multiple generations."""
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(dataset))
    batch = dataset.select(range(start_idx, end_idx))
    prompts = batch["prompt"]
    batch_kwargs = {col: batch[col] for col in extra_columns}

    expanded_prompts = []
    expanded_kwargs = {col: [] for col in extra_columns}
    for i, p in enumerate(prompts):
        for _ in range(num_generations):
            expanded_prompts.append(p)
            for col in extra_columns:
                expanded_kwargs[col].append(batch_kwargs[col][i])

    return expanded_prompts, expanded_kwargs


def _launch_generation(rollout_workers, prompts, config, model_path):
    """Non-blocking: fan out generation across workers, return futures."""
    prompt_chunks = chunk_list(prompts, len(rollout_workers))
    futures = []
    for i, worker in enumerate(rollout_workers):
        if i < len(prompt_chunks) and prompt_chunks[i]:
            futures.append(worker.generate.spawn(
                prompts=prompt_chunks[i],
                model_path=model_path,
                max_tokens=config.generation.max_tokens,
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
                n=1,
                max_model_len=config.model.max_model_len,
            ))
    return futures


def _collect_generation(futures):
    """Blocking: collect results from generation futures."""
    all_completions = []
    all_logprobs = []
    for future in futures:
        result = future.get()
        all_completions.extend(result["completions"])
        all_logprobs.extend(result["logprobs"])
    return all_completions, all_logprobs


def _do_weight_sync(actor, rollout_workers, config, rollout_model_path, current_rollout_model_path, global_step):
    """Perform weight sync using configured method. Returns updated model path."""
    import time
    sync_method = config.training.weight_sync_method
    print(f"Syncing weights (method: {sync_method}, step={global_step}, "
          f"rollout_model_path={rollout_model_path}, n_workers={len(rollout_workers)})...")
    sync_start = time.time()

    try:
        if sync_method == "reload":
            if rollout_model_path is None:
                print("  Warning: rollout_model_path not set, skipping sync")
                return current_rollout_model_path
            actor.sync_weights_to_model_path.remote(
                model_path=rollout_model_path,
                sync_id=global_step + 1,
                config=config.to_dict(),
            )
            sync_futures = [
                worker.update_weights_from_volume.spawn(weights_path=rollout_model_path)
                for worker in rollout_workers
            ]
            sync_results = [f.get() for f in sync_futures]
            success = sum(sync_results)
            print(f"  Weights reloaded in {time.time() - sync_start:.2f}s ({success}/{len(rollout_workers)} workers)")
            return current_rollout_model_path

        elif sync_method == "direct":
            weights_bytes = actor.get_weights.remote(config=config.to_dict())
            sync_futures = [
                worker.update_weights_direct.spawn(weights_bytes)
                for worker in rollout_workers
            ]
            [f.get() for f in sync_futures]
            print(f"  Weights synced directly in {time.time() - sync_start:.2f}s")
            return current_rollout_model_path

        elif sync_method == "volume":
            actor.sync_weights_to_volume.remote(
                sync_id=global_step + 1, config=config.to_dict(),
            )
            sync_futures = [
                worker.load_from_weight_sync.spawn(
                    base_model=config.model.model_name,
                    sync_dir="/storage/weight_sync",
                    max_model_len=config.model.max_model_len,
                )
                for worker in rollout_workers
            ]
            [f.get() for f in sync_futures]
            print(f"  Weights synced via volume in {time.time() - sync_start:.2f}s")
            return current_rollout_model_path

        elif sync_method == "checkpoint":
            checkpoint_path = actor.save_checkpoint.remote(
                step=global_step + 1, config=config.to_dict(),
            )
            sync_futures = [
                worker.reload_from_checkpoint.spawn(checkpoint_path)
                for worker in rollout_workers
            ]
            [f.get() for f in sync_futures]
            print(f"  Weights synced via checkpoint in {time.time() - sync_start:.2f}s")
            return checkpoint_path

        else:
            print(f"  Unknown sync method '{sync_method}', skipping")
            return current_rollout_model_path

    except Exception as e:
        print(f"Warning: Weight sync failed: {e}")
        import traceback
        traceback.print_exc()
        return current_rollout_model_path


# ---------------------------------------------------------------------------
# Async Pipeline Mode
# ---------------------------------------------------------------------------


@app.function(
    image=TRAINING_IMAGE,
    volumes={STORAGE_PATH: volume},
    timeout=3600 * 24,
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
)
def train_async_pipeline(config_dict: Optional[dict] = None, reward_funcs=None, train_dataset=None):
    """Async pipeline: overlap generation N+1 with training N.

    While the actor trains on batch N, rollout workers generate batch N+1.
    This gives ~1-step staleness but significantly reduces GPU idle time.
    """
    import time
    import wandb

    ctx = _setup_distributed(config_dict, reward_funcs, train_dataset)
    config = ctx["config"]
    dataset = ctx["dataset"]
    extra_columns = ctx["extra_columns"]
    actor = ctx["actor"]
    rollout_workers = ctx["rollout_workers"]
    model_path = ctx["current_rollout_model_path"]
    rollout_model_path = ctx["rollout_model_path"]
    reward_funcs = ctx["reward_funcs"]
    total_steps = ctx["total_steps"]
    batch_size = ctx["batch_size"]
    num_batches = ctx["num_batches"]
    num_generations = config.training.num_generations

    async_cfg = config.mode.async_config if hasattr(config.mode, "async_config") else AsyncConfig()
    sync_every = async_cfg.sync_every

    print(f"\n=== ASYNC PIPELINE MODE (sync_every={sync_every}) ===\n")

    # Pre-launch first generation
    prompts_0, kwargs_0 = _get_batch_data(dataset, 0, batch_size, extra_columns, num_generations)
    gen_futures = _launch_generation(rollout_workers, prompts_0, config, model_path)

    global_step = 0
    for epoch in range(config.training.num_epochs):
        for batch_idx in range(num_batches):
            if config.training.max_steps > 0 and global_step >= config.training.max_steps:
                break

            step_start = time.time()

            # Get batch data for this step
            expanded_prompts, expanded_kwargs = _get_batch_data(
                dataset, batch_idx, batch_size, extra_columns, num_generations
            )

            # Collect current generation (blocking)
            gen_start = time.time()
            all_completions, all_logprobs = _collect_generation(gen_futures)
            gen_time = time.time() - gen_start

            print(f"\nStep {global_step + 1}/{total_steps}: "
                  f"{len(all_completions)} completions (gen wait: {gen_time:.1f}s)")

            # Decide if we'll sync weights this step
            will_sync = (global_step + 1) % sync_every == 0
            has_next = (batch_idx + 1 < num_batches and not (
                config.training.max_steps > 0 and global_step + 1 >= config.training.max_steps
            ))

            print(f"  [pipeline] will_sync={will_sync}, has_next={has_next}")

            # Launch NEXT generation — but NOT if we need to sync weights,
            # because the rollout worker can't handle concurrent method calls
            # (Modal would spin up a new replica with uninitialized state).
            if has_next and not will_sync:
                next_prompts, _ = _get_batch_data(
                    dataset, batch_idx + 1, batch_size, extra_columns, num_generations
                )
                print(f"  [pipeline] Pre-launching next generation (overlapping with train)")
                gen_futures = _launch_generation(rollout_workers, next_prompts, config, model_path)
            else:
                if will_sync:
                    print(f"  [pipeline] Skipping pre-launch: weight sync needed (worker must be idle)")
                gen_futures = []

            # Compute rewards + train (runs WHILE next generation happens)
            train_start = time.time()
            wrapped = [[{"role": "assistant", "content": c}] for c in all_completions]
            rewards = compute_rewards(
                reward_funcs=reward_funcs, completions=wrapped,
                prompts=expanded_prompts, **expanded_kwargs,
            )
            mean_reward = sum(rewards) / len(rewards) if rewards else 0

            loss_result = actor.train_step.remote(
                prompts=expanded_prompts, completions=all_completions,
                rewards=rewards, old_logprobs=all_logprobs,
                config=config.to_dict(),
            )
            train_time = time.time() - train_start

            # Metrics
            step_time = time.time() - step_start
            overlap = gen_time / step_time if step_time > 0 else 0
            metrics = {
                "train/loss": loss_result.get("loss", 0),
                "train/mean_reward": mean_reward,
                "train/approx_kl": loss_result.get("approx_kl", 0),
                "train/clip_fraction": loss_result.get("clip_fraction", 0),
                "async/gen_time_s": gen_time,
                "async/train_time_s": train_time,
                "async/step_time_s": step_time,
                "async/overlap_ratio": overlap,
                "async/mode": 0,  # 0=pipeline
            }
            wandb.log(metrics, step=global_step)
            print(f"  Loss: {loss_result.get('loss', 0):.4f}, "
                  f"Reward: {mean_reward:.4f}, "
                  f"Step: {step_time:.1f}s (gen={gen_time:.1f}s, train={train_time:.1f}s)")

            # Weight sync — rollout workers are idle (no pre-launched generation)
            if will_sync:
                print(f"  [pipeline] Starting weight sync (rollout workers should be idle)")
                model_path = _do_weight_sync(
                    actor, rollout_workers, config,
                    rollout_model_path, model_path, global_step,
                )

                # Now launch next generation with UPDATED weights
                if has_next:
                    next_prompts, _ = _get_batch_data(
                        dataset, batch_idx + 1, batch_size, extra_columns, num_generations
                    )
                    print(f"  [pipeline] Post-sync: launching next gen with updated weights")
                    gen_futures = _launch_generation(rollout_workers, next_prompts, config, model_path)

            # Checkpoint
            if (global_step + 1) % config.training.save_steps == 0:
                actor.save_checkpoint.remote(global_step + 1, config=config.to_dict())

            global_step += 1

    # Final checkpoint
    final_checkpoint = actor.save_checkpoint.remote(global_step, config=config.to_dict())
    volume.commit()
    wandb.finish()

    return {"status": "completed", "total_steps": global_step, "final_checkpoint": final_checkpoint}


# ---------------------------------------------------------------------------
# Async Queue Mode
# ---------------------------------------------------------------------------


@app.function(
    image=TRAINING_IMAGE,
    volumes={STORAGE_PATH: volume},
    timeout=3600 * 24,
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
)
def train_async_queue(config_dict: Optional[dict] = None, reward_funcs=None, train_dataset=None):
    """Async queue: fully decoupled producer-consumer via modal.Queue.

    A separate _rollout_producer function continuously generates batches
    into a modal.Queue. This function consumes batches and trains.
    """
    import time
    import wandb

    ctx = _setup_distributed(config_dict, reward_funcs, train_dataset)
    config = ctx["config"]
    actor = ctx["actor"]
    rollout_workers = ctx["rollout_workers"]
    model_path = ctx["current_rollout_model_path"]
    rollout_model_path = ctx["rollout_model_path"]
    reward_funcs = ctx["reward_funcs"]
    total_steps = ctx["total_steps"]

    async_cfg = config.mode.async_config if hasattr(config.mode, "async_config") else AsyncConfig()
    staleness_threshold = async_cfg.staleness_threshold
    sync_every = async_cfg.sync_every
    queue_size = async_cfg.queue_size

    print(f"\n=== ASYNC QUEUE MODE (queue_size={queue_size}, "
          f"staleness_threshold={staleness_threshold}, sync_every={sync_every}) ===\n")

    with modal.Queue.ephemeral() as queue:
        # Launch producer as separate Modal function
        producer_handle = _rollout_producer.spawn(
            queue=queue,
            config_dict=config_dict,
            train_dataset=train_dataset,
            model_path=model_path,
            rollout_model_path=rollout_model_path,
            max_queue_size=queue_size,
        )

        # Consumer loop
        param_version = 0
        global_step = 0
        stale_drops = 0

        while global_step < total_steps:
            step_start = time.time()

            # Get batch from queue (blocking with timeout)
            batch = queue.get(partition="batches", timeout=600)
            if batch is None:
                print(f"Queue timeout at step {global_step} — producer may have stopped")
                break

            # Staleness check
            batch_version = batch.get("param_version", 0)
            staleness = param_version - batch_version
            if staleness > staleness_threshold:
                stale_drops += 1
                print(f"  Dropping stale batch (staleness={staleness}, threshold={staleness_threshold})")
                continue

            all_completions = batch["completions"]
            all_logprobs = batch["logprobs"]
            expanded_prompts = batch["prompts"]
            expanded_kwargs = batch.get("kwargs", {})

            print(f"\nStep {global_step + 1}/{total_steps}: "
                  f"{len(all_completions)} completions (staleness={staleness})")

            # Rewards + train
            train_start = time.time()
            wrapped = [[{"role": "assistant", "content": c}] for c in all_completions]
            rewards = compute_rewards(
                reward_funcs=reward_funcs, completions=wrapped,
                prompts=expanded_prompts, **expanded_kwargs,
            )
            mean_reward = sum(rewards) / len(rewards) if rewards else 0

            loss_result = actor.train_step.remote(
                prompts=expanded_prompts, completions=all_completions,
                rewards=rewards, old_logprobs=all_logprobs,
                config=config.to_dict(),
            )
            train_time = time.time() - train_start
            param_version += 1

            # Metrics
            queue_depth = queue.len(partition="batches")
            metrics = {
                "train/loss": loss_result.get("loss", 0),
                "train/mean_reward": mean_reward,
                "train/approx_kl": loss_result.get("approx_kl", 0),
                "train/clip_fraction": loss_result.get("clip_fraction", 0),
                "async/staleness": staleness,
                "async/queue_depth": queue_depth,
                "async/param_version": param_version,
                "async/train_time_s": train_time,
                "async/stale_drops": stale_drops,
                "async/mode": 1,  # 1=queue
            }
            wandb.log(metrics, step=global_step)
            print(f"  Loss: {loss_result.get('loss', 0):.4f}, "
                  f"Reward: {mean_reward:.4f}, "
                  f"Queue depth: {queue_depth}, "
                  f"Train: {train_time:.1f}s")

            # Weight sync — save weights, signal producer (non-blocking)
            if param_version % sync_every == 0:
                print(f"  [queue] Saving actor weights for sync (step={global_step})...")
                actor.sync_weights_to_model_path.remote(
                    model_path=rollout_model_path,
                    sync_id=global_step + 1,
                    config=config.to_dict(),
                )
                # Tell producer to reload weights after its current generation finishes
                # This is non-blocking — producer picks it up when it's ready
                queue.put({
                    "sync_weights": True,
                    "version": param_version,
                    "model_path": rollout_model_path,
                }, partition="control")
                model_path = rollout_model_path
                print(f"  [queue] Signaled producer to sync weights (v={param_version})")

            # Checkpoint
            if (global_step + 1) % config.training.save_steps == 0:
                actor.save_checkpoint.remote(global_step + 1, config=config.to_dict())

            global_step += 1

        # Signal producer to stop
        queue.put({"stop": True}, partition="control")
        print(f"\nTraining complete. Waiting for producer to finish...")

        # Final checkpoint
        final_checkpoint = actor.save_checkpoint.remote(global_step, config=config.to_dict())
        volume.commit()
        wandb.finish()

    return {"status": "completed", "total_steps": global_step, "final_checkpoint": final_checkpoint}


@app.function(
    image=TRAINING_IMAGE,
    volumes={STORAGE_PATH: volume},
    timeout=3600 * 24,
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
)
def _rollout_producer(queue, config_dict, train_dataset, model_path, rollout_model_path, max_queue_size=2):
    """Continuously generates batches into a modal.Queue.

    Runs as a separate Modal function from the consumer (train_async_queue).
    Uses its own rollout workers to generate completions.
    """
    import time
    from datasets import load_dataset
    from transformers import AutoTokenizer

    config = OrchestratorConfig.from_dict(config_dict or {})
    rollout_gpu = config.rollout_gpu

    # Initialize rollout workers (separate from consumer's)
    print(f"[Producer] Initializing {config.num_rollout_workers} rollout workers...")
    rollout_workers = [
        RolloutWorker.with_options(gpu=rollout_gpu)()
        for _ in range(config.num_rollout_workers)
    ]

    # Pre-warm
    sync_method = config.training.weight_sync_method
    current_model_path = model_path

    if sync_method == "reload" and rollout_model_path:
        warmup_futures = []
        for worker in rollout_workers:
            future = worker.initialize_for_weight_sync.spawn(
                base_model=config.model.model_name,
                max_model_len=config.model.max_model_len,
            )
            warmup_futures.append(future)
        for future in warmup_futures:
            future.get()
        warmup_gen_futures = []
        for worker in rollout_workers:
            future = worker.generate.spawn(
                prompts=["Hello"], model_path=current_model_path,
                max_tokens=10, max_model_len=config.model.max_model_len,
            )
            warmup_gen_futures.append(future)
        for f in warmup_gen_futures:
            f.get()
    else:
        warmup_futures = []
        for worker in rollout_workers:
            future = worker.generate.spawn(
                prompts=["Hello"], model_path=current_model_path,
                max_tokens=10, max_model_len=config.model.max_model_len,
            )
            warmup_futures.append(future)
        for f in warmup_futures:
            f.get()

    print("[Producer] Rollout workers warmed up")

    # Load dataset
    if train_dataset is not None:
        if callable(train_dataset):
            print("[Producer] Running dataset prep function on container...")
            dataset = train_dataset()
        else:
            dataset = train_dataset
        if config.max_samples:
            dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    else:
        dataset = load_dataset(
            config.dataset_name, config.dataset_config, split=config.dataset_split,
        )
        dataset = dataset.rename_column("instruction", "prompt")
        dataset = dataset.rename_column("testcase", "testcases")
        if config.max_samples:
            dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    # Chat template
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, trust_remote_code=True)
    sample_prompt = dataset[0]["prompt"]
    if isinstance(sample_prompt, list) and len(sample_prompt) > 0 and isinstance(sample_prompt[0], dict):
        def apply_chat_template(example):
            example["prompt"] = tokenizer.apply_chat_template(
                example["prompt"], tokenize=False, add_generation_prompt=True,
            )
            return example
        dataset = dataset.map(apply_chat_template)

    extra_columns = [c for c in dataset.column_names if c != "prompt"]
    batch_size = config.training.batch_size
    num_generations = config.training.num_generations
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    param_version = 0
    batch_idx = 0

    print(f"[Producer] Starting continuous generation loop (queue_size={max_queue_size})...")

    def _process_control_messages():
        """Process all pending control messages. Returns True if should stop."""
        nonlocal param_version, current_model_path
        while True:
            ctrl = queue.get(partition="control", block=False)
            if ctrl is None:
                return False
            if ctrl.get("stop"):
                print("[Producer] Received stop signal")
                return True
            if ctrl.get("sync_weights"):
                # Workers are idle (generation already collected), reload weights
                new_path = ctrl.get("model_path", current_model_path)
                new_version = ctrl.get("version", param_version)
                print(f"[Producer] Syncing weights: v={new_version}, path={new_path}")
                # Workers reload via vLLM sleep/wake/reload_weights — no volume.reload() needed
                # (volume.reload() fails with "open files" because vLLM has model files open)
                try:
                    for worker in rollout_workers:
                        worker.update_weights_from_volume.remote(weights_path=new_path)
                    current_model_path = new_path
                    param_version = new_version
                    print(f"[Producer] Weight sync done (v={param_version})")
                except Exception as e:
                    print(f"[Producer] Weight sync failed (likely shutdown): {e}")
                continue
            if "version" in ctrl:
                param_version = ctrl["version"]
                print(f"[Producer] Updated param_version to {param_version}")
            if "model_path" in ctrl:
                new_path = ctrl["model_path"]
                if new_path != current_model_path:
                    current_model_path = new_path
                    print(f"[Producer] Updated model_path to {current_model_path}")

    while True:
        # Check for control messages (non-blocking)
        if _process_control_messages():
            break

        # Backpressure: pause if queue is full
        depth = queue.len(partition="batches")
        if depth >= max_queue_size:
            time.sleep(0.5)
            continue

        # Wrap around dataset
        actual_batch_idx = batch_idx % num_batches

        # Generate
        expanded_prompts, expanded_kwargs = _get_batch_data(
            dataset, actual_batch_idx, batch_size, extra_columns, num_generations,
        )

        gen_start = time.time()
        gen_futures = _launch_generation(rollout_workers, expanded_prompts, config, current_model_path)
        try:
            all_completions, all_logprobs = _collect_generation(gen_futures)
        except Exception as e:
            print(f"[Producer] Generation failed (likely shutdown): {e}")
            break
        gen_time = time.time() - gen_start

        # Check for control messages AFTER generation completes (workers now idle)
        if _process_control_messages():
            break

        # Put into queue
        queue.put({
            "prompts": expanded_prompts,
            "completions": all_completions,
            "logprobs": all_logprobs,
            "kwargs": expanded_kwargs,
            "param_version": param_version,
        }, partition="batches")

        print(f"[Producer] Batch {batch_idx} → queue (v={param_version}, "
              f"{len(all_completions)} completions, {gen_time:.1f}s, depth={depth + 1})")

        batch_idx += 1

    print("[Producer] Exiting")
