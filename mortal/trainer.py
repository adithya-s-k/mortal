"""MortalTrainer - TRL-like programmatic API for MORTAL training."""

from mortal.config import OrchestratorConfig
from mortal.rewards.base import RewardEnvironment


class MortalTrainer:
    """High-level trainer facade for MORTAL.

    Provides a TRL-like API for launching GRPO training on Modal.
    Automatically routes to remote (Modal orchestrator) or local orchestrator
    depending on whether custom reward functions are provided.

    Examples:
        # Sandbox reward (existing behavior, orchestrator runs on Modal)
        trainer = MortalTrainer(model="Qwen/Qwen2.5-0.5B-Instruct")
        trainer.train()

        # Custom reward (orchestrator runs locally, GPU work on Modal)
        def length_reward(completions, **kwargs):
            return [min(len(c) / 100, 1.0) for c in completions]

        trainer = MortalTrainer(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            reward_funcs=length_reward,
            train_dataset=dataset,
            max_steps=5,
        )
        trainer.train()

        # RewardEnvironment (code execution with custom sandbox)
        from mortal.rewards.examples import CodeExecutionEnvironment
        from mortal.rewards import SandboxConfig
        env = CodeExecutionEnvironment(
            partial_credit=True,
            sandbox_cfg=SandboxConfig(image=modal.Image.debian_slim().pip_install("numpy")),
        )
        trainer = MortalTrainer(reward_funcs=env, train_dataset=ds)
        trainer.train()

        # Mixed rewards with weights
        trainer = MortalTrainer(
            reward_funcs=[env, length_reward],
            reward_weights=[0.7, 0.3],
            train_dataset=ds,
        )
        trainer.train()
    """

    def __init__(
        self,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=None,
        reward_weights=None,
        train_dataset=None,
        num_generations=4,
        learning_rate=5e-6,
        batch_size=8,
        num_epochs=5,
        max_steps=-1,
        loss_type="dapo",
        actor_gpu="A100",
        rollout_gpu="A10G",
        num_rollout_workers=2,
        max_samples=128,
        max_tokens=8000,
        max_model_len=16384,
        max_completion_length=1024,
        weight_sync_method="reload",
        sync_weights_every=1,
        save_steps=100,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=None,
        scale_rewards="group",
        mask_truncated_completions=False,
        use_lora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        gradient_checkpointing=True,
        dataset_name="OpenCoder-LLM/opc-sft-stage2",
        dataset_config="educational_instruct",
        dataset_split="train",
        **kwargs,
    ):
        """Initialize MortalTrainer.

        Args:
            model: Model name or HuggingFace path.
            reward_funcs: Reward function(s). One of:
                - None or "sandbox": uses Modal Sandbox code execution (default)
                - RewardEnvironment: custom environment with score() method
                - callable: custom reward function(completions=..., **kwargs)
                - list of the above: multiple rewards, weighted average
            reward_weights: Optional list of weights for combining multiple
                reward functions. If None, uses RewardEnvironment.weight
                for environments and 1.0 for callables.
            train_dataset: HuggingFace Dataset with "prompt" column.
                Required for custom reward_funcs, optional for sandbox mode.
            num_generations: Number of generations per prompt for GRPO.
            learning_rate: Learning rate.
            batch_size: Batch size.
            num_epochs: Number of training epochs.
            max_steps: Maximum training steps (-1 for unlimited).
            loss_type: GRPO loss type (grpo, dr_grpo, dapo, bnpo, cispo, sapo).
            actor_gpu: GPU type for actor worker.
            rollout_gpu: GPU type for rollout workers.
            num_rollout_workers: Number of vLLM rollout workers.
            max_samples: Maximum dataset samples (None for full dataset).
            max_tokens: Maximum tokens per generated completion.
            max_model_len: Maximum model context length.
            max_completion_length: Max completion length for training.
            weight_sync_method: Weight sync method (reload, volume, direct, checkpoint).
            sync_weights_every: Sync weights every N steps.
            save_steps: Save checkpoint every N steps.
            beta: KL penalty coefficient.
            epsilon: PPO-style clipping epsilon.
            epsilon_high: Upper clipping epsilon (None to disable).
            scale_rewards: Reward scaling (group, batch, none).
            mask_truncated_completions: Mask truncated completions.
            use_lora: Enable LoRA.
            lora_r: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_dropout: LoRA dropout.
            gradient_checkpointing: Enable gradient checkpointing.
            dataset_name: HuggingFace dataset name (for sandbox mode).
            dataset_config: Dataset config name.
            dataset_split: Dataset split.
            **kwargs: Additional config overrides.
        """
        self.reward_funcs = reward_funcs
        self.reward_weights = reward_weights
        self.train_dataset = train_dataset

        config_dict = {
            "model_name": model,
            "num_generations": num_generations,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "max_steps": max_steps,
            "loss_type": loss_type,
            "actor_gpu": actor_gpu,
            "rollout_gpu": rollout_gpu,
            "num_rollout_workers": num_rollout_workers,
            "max_samples": max_samples,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "max_completion_length": max_completion_length,
            "weight_sync_method": weight_sync_method,
            "sync_weights_every": sync_weights_every,
            "save_steps": save_steps,
            "beta": beta,
            "epsilon": epsilon,
            "epsilon_high": epsilon_high,
            "scale_rewards": scale_rewards,
            "mask_truncated_completions": mask_truncated_completions,
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "gradient_checkpointing": gradient_checkpointing,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
        }
        config_dict.update(kwargs)

        self.config = OrchestratorConfig.from_dict(config_dict)

    def train(self):
        """Run training.

        Routes to the appropriate backend:
        - Remote (Modal orchestrator) for sandbox rewards
        - Local orchestrator for custom reward functions

        Both paths require a Modal App context (for GPU workers), so
        we wrap in app.run() to hydrate the app when called from a
        regular Python script (outside `modal run`).
        """
        import modal
        from mortal.app import app

        # Import orchestrator (and transitively all workers) so that
        # app.run() discovers and hydrates all Modal functions/classes.
        from mortal import orchestrator as _orch  # noqa: F841
        import mortal.rewards.function_executor  # noqa: F401  — hydrate _run_on_training_image

        with modal.enable_output():
            with app.run():
                if self.reward_funcs is None or self.reward_funcs == "sandbox":
                    print("Using remote orchestrator (sandbox rewards on Modal)")
                    return _orch.train.remote(self.config.to_dict())
                else:
                    # RewardEnvironment, callable, or list -> local orchestrator
                    print("Using local orchestrator (custom rewards)")
                    return _orch.train_local(
                        self.config.to_dict(),
                        self.reward_funcs,
                        self.train_dataset,
                        self.reward_weights,
                    )
