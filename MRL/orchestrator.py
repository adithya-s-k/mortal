"""Main training orchestrator - veRL style coordination of workers."""

from typing import Optional

import modal

from MRL.app import app, volume, TRAINING_IMAGE
from MRL.config import OrchestratorConfig
from MRL.workers.actor import ActorWorker
from MRL.workers.rollout import RolloutWorker
from MRL.rewards import compute_rewards

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
def train(config_dict: Optional[dict] = None, reward_funcs=None):
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
            batch = get_batch(dataset, batch_idx, batch_size)
            prompts = batch["prompts"]
            testcases = batch["testcases"]

            print(
                f"\nStep {global_step + 1}/{total_steps}: Processing {len(prompts)} prompts"
            )

            # 2. Generate completions (parallel across rollout workers)
            # Expand prompts for multiple generations per prompt
            expanded_prompts = []
            expanded_testcases = []
            for p, t in zip(prompts, testcases):
                for _ in range(config.training.num_generations):
                    expanded_prompts.append(p)
                    expanded_testcases.append(t)

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
            print("Computing rewards...")
            rewards = compute_rewards(
                reward_funcs=reward_funcs,
                completions=all_completions,
                testcases=expanded_testcases,
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


@app.function(
    image=TRAINING_IMAGE,
    gpu="A100",
    volumes={STORAGE_PATH: volume},
    timeout=3600 * 24,
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
)
def train_simple(config_dict: Optional[dict] = None):
    """Simple training using TRL's built-in trainer loop.

    This is a simpler alternative that uses GRPOTrainer's internal
    training loop instead of manual orchestration. Useful for debugging
    or when the manual orchestration overhead is not needed.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Training result summary
    """
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer
    import torch

    from MRL.workers.reward import reward_helper_function

    # Parse config
    if config_dict is None:
        config_dict = {}
    config = OrchestratorConfig.from_dict(config_dict)

    print(f"Starting simple training with config: {config.to_dict()}")

    # Load dataset
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

    # Build GRPOConfig kwargs
    grpo_kwargs = {
        "output_dir": f"{STORAGE_PATH}/checkpoints",
        "report_to": config.training.report_to,
        "use_vllm": False,
        "per_device_train_batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "num_train_epochs": config.training.num_epochs,
        "max_steps": config.training.max_steps,
        "save_steps": config.training.save_steps,
        "logging_steps": config.training.logging_steps,
        "num_generations": config.training.num_generations,
        "bf16": torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        # GRPO algorithm parameters (TRL built-in)
        "loss_type": config.training.loss_type,
        "beta": config.training.beta,
        "epsilon": config.training.epsilon,
        "scale_rewards": config.training.scale_rewards,
        "mask_truncated_completions": config.training.mask_truncated_completions,
    }

    # Only add epsilon_high if specified
    if config.training.epsilon_high is not None:
        grpo_kwargs["epsilon_high"] = config.training.epsilon_high

    training_args = GRPOConfig(**grpo_kwargs)

    # Configure LoRA if enabled
    peft_config = None
    if config.training.use_lora:
        from peft import LoraConfig, TaskType

        target_modules = config.training.lora_target_modules
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        peft_config = LoraConfig(
            r=config.training.lora_r,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=config.training.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        print(f"LoRA enabled: r={peft_config.r}, alpha={peft_config.lora_alpha}")

    # Create trainer
    trainer_kwargs = {
        "model": config.model.model_name,
        "reward_funcs": reward_helper_function,
        "args": training_args,
        "train_dataset": dataset,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)

    # Train
    trainer.train()

    # Commit volume
    volume.commit()

    return {"status": "completed"}


def train_local(config_dict: dict, reward_funcs, train_dataset=None):
    """Local orchestrator - runs on user's CPU with custom reward functions.

    Same training loop as train() but:
    - Runs locally (not as a Modal function)
    - Dataset loaded locally or from train_dataset arg
    - Rewards computed locally via compute_rewards()
    - GPU workers (actor, rollout) still run on Modal via .remote()

    Args:
        config_dict: Configuration dictionary
        reward_funcs: Custom reward function(s) (callable or list of callables)
        train_dataset: Optional HuggingFace Dataset with "prompt" column.
            If None, loads from config dataset settings.
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
