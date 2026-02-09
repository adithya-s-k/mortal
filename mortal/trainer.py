"""MortalTrainer - TRL-like programmatic API for MORTAL training."""

from mortal.config import OrchestratorConfig, SingleNode, Distributed, GPUConfig
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
        mode=None,
        reward_funcs=None,
        reward_weights=None,
        train_dataset=None,
        num_generations=4,
        learning_rate=5e-6,
        batch_size=8,
        num_epochs=5,
        max_steps=-1,
        loss_type="dapo",
        max_samples=128,
        max_tokens=8000,
        max_model_len=16384,
        max_completion_length=1024,
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
        # Legacy params (deprecated, use mode= instead)
        actor_gpu=None,
        rollout_gpu=None,
        num_rollout_workers=None,
        weight_sync_method=None,
        sync_weights_every=None,
        **kwargs,
    ):
        """Initialize MortalTrainer.

        Args:
            model: Model name or HuggingFace path.
            mode: Execution mode. One of:
                - SingleNode(...): Single container, optional vLLM.
                - Distributed(...): veRL-style separate actor + rollout workers.
                - None: defaults to Distributed().
            reward_funcs: Reward function(s). One of:
                - None or "sandbox": uses Modal Sandbox code execution (default)
                - RewardEnvironment: custom environment with score() method
                - callable: custom reward function(completions=..., **kwargs)
                - list of the above: multiple rewards, weighted average
            reward_weights: Optional list of weights for combining multiple rewards.
            train_dataset: HuggingFace Dataset with "prompt" column, or a
                callable that returns one (runs on container, no local loading).
            **kwargs: Additional config overrides.
        """
        self.reward_funcs = reward_funcs
        self.reward_weights = reward_weights
        self.train_dataset = train_dataset

        # Build mode object
        if mode is None:
            # Support legacy params
            mode_kwargs = {}
            if actor_gpu is not None:
                mode_kwargs["actor"] = actor_gpu
            if rollout_gpu is not None:
                mode_kwargs["rollout"] = rollout_gpu
            if num_rollout_workers is not None:
                mode_kwargs["num_rollout_workers"] = num_rollout_workers
            if weight_sync_method is not None:
                mode_kwargs["weight_sync_method"] = weight_sync_method
            if sync_weights_every is not None:
                mode_kwargs["sync_weights_every"] = sync_weights_every
            mode = Distributed(**mode_kwargs) if mode_kwargs else Distributed()

        config_dict = {
            "model_name": model,
            "num_generations": num_generations,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "max_steps": max_steps,
            "loss_type": loss_type,
            "max_samples": max_samples,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "max_completion_length": max_completion_length,
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
            "mode": mode.to_dict(),
        }
        config_dict.update(kwargs)

        self.config = OrchestratorConfig.from_dict(config_dict)

    def train(self, detach=False):
        """Run training.

        Args:
            detach: If True, the training runs on Modal independently of the
                local process. The terminal can be closed and training continues.
                If False (default), blocks until training completes.

        Routes to the appropriate backend:
        - Remote (Modal orchestrator) for sandbox rewards
        - Local orchestrator for custom reward functions
        """
        import modal
        from mortal.app import app

        # Import orchestrator (and transitively all workers) so that
        # app.run() discovers and hydrates all Modal functions/classes.
        from mortal import orchestrator as _orch  # noqa: F841
        import mortal.rewards.function_executor  # noqa: F401  — hydrate _run_on_training_image

        with modal.enable_output():
            with app.run(detach=detach):
                mode = self.config.mode

                if isinstance(mode, SingleNode):
                    gpu_spec = mode.gpu.to_modal_spec()
                    print(f"Using single-node training (gpu={gpu_spec}, "
                          f"use_vllm={mode.use_vllm}, vllm_mode={mode.vllm_mode})")
                    worker = _orch.SingleNodeTrainer.with_options(gpu=gpu_spec)()
                    reward_funcs = self.reward_funcs if self.reward_funcs not in [None, "sandbox"] else None
                    # train_dataset can be:
                    #   - callable: runs on container (no local data loading)
                    #   - Dataset: serialize to in-memory for transfer
                    #   - None: load from config on container
                    train_ds = self.train_dataset
                    if train_ds is not None and not callable(train_ds):
                        from datasets import Dataset
                        train_ds = Dataset.from_dict(train_ds.to_dict())
                    run_kwargs = dict(
                        config_dict=self.config.to_dict(),
                        reward_funcs=reward_funcs,
                        train_dataset=train_ds,
                    )
                    if detach:
                        worker.run.spawn(**run_kwargs)
                        print("Training spawned in detached mode. Check Modal dashboard for progress.")
                        return None
                    return worker.run.remote(**run_kwargs)

                elif isinstance(mode, Distributed):
                    print("Using remote orchestrator (distributed on Modal)")
                    reward_funcs = self.reward_funcs if self.reward_funcs not in [None, "sandbox"] else None
                    train_ds = self.train_dataset
                    if train_ds is not None and not callable(train_ds):
                        from datasets import Dataset
                        train_ds = Dataset.from_dict(train_ds.to_dict())
                    if detach:
                        _orch.train.spawn(self.config.to_dict(), reward_funcs=reward_funcs, train_dataset=train_ds)
                        print("Training spawned in detached mode. Check Modal dashboard for progress.")
                        return None
                    return _orch.train.remote(self.config.to_dict(), reward_funcs=reward_funcs, train_dataset=train_ds)

                else:
                    raise ValueError(f"Unknown mode type: {type(mode)}")
