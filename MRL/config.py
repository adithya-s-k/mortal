"""Configuration dataclasses for GRPO training."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    max_model_len: int = 16384
    trust_remote_code: bool = True


@dataclass
class GenerationConfig:
    """Generation configuration for rollout workers."""

    max_tokens: int = 8000
    temperature: float = 0.7
    top_p: float = 0.9
    n: int = 1  # Number of completions per prompt


@dataclass
class TrainingConfig:
    """Training configuration for the actor worker."""

    # Basic training
    num_epochs: int = 5
    max_steps: int = -1  # -1 means use epochs
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6

    # GRPO specific
    num_generations: int = 4  # Number of generations per prompt for GRPO

    # GRPO algorithm parameters (passed to TRL's GRPOConfig)
    # Loss type: grpo, dr_grpo, dapo, bnpo, cispo, sapo
    loss_type: str = "dapo"
    # KL coefficient (DeepSeek R1 uses 0.001)
    beta: float = 0.0
    # PPO-style clipping epsilon
    epsilon: float = 0.2
    # Upper clipping epsilon (DAPO recommends 0.28, None to disable)
    epsilon_high: Optional[float] = None
    # Reward scaling: "group", "batch", or "none"
    scale_rewards: str = "group"
    # Whether to mask truncated completions (DAPO paper)
    mask_truncated_completions: bool = False
    # Max completion length for training (truncates long completions to save memory)
    max_completion_length: int = 1024

    # LoRA parameters
    use_lora: bool = False
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha (scaling factor)
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list] = None  # None means auto-detect

    # Memory optimizations
    gradient_checkpointing: bool = True  # Reduces memory by recomputing activations

    # Legacy (kept for backwards compatibility, use beta instead)
    kl_coef: float = 0.1

    # Checkpointing
    save_steps: int = 100
    checkpoint_dir: str = "/storage/checkpoints"

    # Logging
    report_to: str = "wandb"
    logging_steps: int = 10

    # Weight sync
    sync_weights_every: int = 1  # Sync weights to rollout workers every N steps
    # Sync method:
    #   "reload" (recommended) - uses vLLM v1 sleep/wake_up/reload_weights for efficient updates
    #   "volume" - saves to shared volume, workers reload from volume (recreates model)
    #   "direct" - in-memory transfer via vLLM's load_weights()
    #   "checkpoint" - full checkpoint save + model recreation (slowest)
    weight_sync_method: str = "reload"


@dataclass
class OrchestratorConfig:
    """Configuration for the training orchestrator."""

    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Generation
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # Workers
    num_rollout_workers: int = 2

    # GPU allocation
    actor_gpu: str = "A100"    # "A100", "H100", "A10G", etc.
    rollout_gpu: str = "A10G"  # "A10G", "A100", "L4", etc.

    # Data
    dataset_name: str = "OpenCoder-LLM/opc-sft-stage2"
    dataset_config: str = "educational_instruct"
    dataset_split: str = "train"
    max_samples: Optional[int] = 128  # None for full dataset

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model_name": self.model.model_name,
            "max_model_len": self.model.max_model_len,
            "trust_remote_code": self.model.trust_remote_code,
            "max_tokens": self.generation.max_tokens,
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "n": self.generation.n,
            "num_epochs": self.training.num_epochs,
            "max_steps": self.training.max_steps,
            "batch_size": self.training.batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "learning_rate": self.training.learning_rate,
            "num_generations": self.training.num_generations,
            "loss_type": self.training.loss_type,
            "beta": self.training.beta,
            "epsilon": self.training.epsilon,
            "epsilon_high": self.training.epsilon_high,
            "scale_rewards": self.training.scale_rewards,
            "mask_truncated_completions": self.training.mask_truncated_completions,
            "max_completion_length": self.training.max_completion_length,
            "use_lora": self.training.use_lora,
            "lora_r": self.training.lora_r,
            "lora_alpha": self.training.lora_alpha,
            "lora_dropout": self.training.lora_dropout,
            "lora_target_modules": self.training.lora_target_modules,
            "gradient_checkpointing": self.training.gradient_checkpointing,
            "kl_coef": self.training.kl_coef,
            "save_steps": self.training.save_steps,
            "checkpoint_dir": self.training.checkpoint_dir,
            "report_to": self.training.report_to,
            "logging_steps": self.training.logging_steps,
            "sync_weights_every": self.training.sync_weights_every,
            "weight_sync_method": self.training.weight_sync_method,
            "num_rollout_workers": self.num_rollout_workers,
            "actor_gpu": self.actor_gpu,
            "rollout_gpu": self.rollout_gpu,
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "dataset_split": self.dataset_split,
            "max_samples": self.max_samples,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrchestratorConfig":
        """Create config from dictionary."""
        model = ModelConfig(
            model_name=d.get("model_name", "Qwen/Qwen2-0.5B-Instruct"),
            max_model_len=d.get("max_model_len", 16384),
            trust_remote_code=d.get("trust_remote_code", True),
        )
        generation = GenerationConfig(
            max_tokens=d.get("max_tokens", 8000),
            temperature=d.get("temperature", 0.7),
            top_p=d.get("top_p", 0.9),
            n=d.get("n", 1),
        )
        training = TrainingConfig(
            num_epochs=d.get("num_epochs", 5),
            max_steps=d.get("max_steps", -1),
            batch_size=d.get("batch_size", 8),
            gradient_accumulation_steps=d.get("gradient_accumulation_steps", 1),
            learning_rate=d.get("learning_rate", 5e-6),
            num_generations=d.get("num_generations", 4),
            loss_type=d.get("loss_type", "dapo"),
            beta=d.get("beta", 0.0),
            epsilon=d.get("epsilon", 0.2),
            epsilon_high=d.get("epsilon_high"),
            scale_rewards=d.get("scale_rewards", "group"),
            mask_truncated_completions=d.get("mask_truncated_completions", False),
            max_completion_length=d.get("max_completion_length", 1024),
            use_lora=d.get("use_lora", False),
            lora_r=d.get("lora_r", 16),
            lora_alpha=d.get("lora_alpha", 32),
            lora_dropout=d.get("lora_dropout", 0.05),
            lora_target_modules=d.get("lora_target_modules"),
            gradient_checkpointing=d.get("gradient_checkpointing", True),
            kl_coef=d.get("kl_coef", 0.1),
            save_steps=d.get("save_steps", 100),
            checkpoint_dir=d.get("checkpoint_dir", "/storage/checkpoints"),
            report_to=d.get("report_to", "wandb"),
            logging_steps=d.get("logging_steps", 10),
            sync_weights_every=d.get("sync_weights_every", 1),
            weight_sync_method=d.get("weight_sync_method", "reload"),
        )
        return cls(
            model=model,
            generation=generation,
            training=training,
            num_rollout_workers=d.get("num_rollout_workers", 2),
            actor_gpu=d.get("actor_gpu", "A100"),
            rollout_gpu=d.get("rollout_gpu", "A10G"),
            dataset_name=d.get("dataset_name", "OpenCoder-LLM/opc-sft-stage2"),
            dataset_config=d.get("dataset_config", "educational_instruct"),
            dataset_split=d.get("dataset_split", "train"),
            max_samples=d.get("max_samples", 128),
        )
