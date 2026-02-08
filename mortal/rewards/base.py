"""Base classes for the reward environments system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxConfig:
    """Configuration for Modal Sandbox execution.

    Sandboxes create a fresh isolated container per call. Best for
    untrusted code (e.g. running student-generated code).

    Attributes:
        image: A modal.Image object. None uses modal.Image.debian_slim().
        timeout: Max execution time in seconds.
        secrets: List of Modal secret names to mount.

    Example:
        import modal
        cfg = SandboxConfig(
            image=modal.Image.debian_slim().pip_install("numpy", "pandas"),
            timeout=60,
        )
    """

    image: Any = None  # modal.Image — typed as Any to avoid import at definition time
    timeout: int = 30
    secrets: list[str] = field(default_factory=list)


@dataclass
class FunctionConfig:
    """Configuration for Modal Function execution.

    Functions run in pre-warmed containers. Faster than sandboxes,
    support GPU. Best for trusted computation.

    Attributes:
        image: A modal.Image object. None uses TRAINING_IMAGE
            (has torch/trl/numpy/transformers pre-installed).
        gpu: GPU type string or None for CPU-only. E.g. "T4", "A10G", "A100".
        timeout: Max execution time in seconds.
        secrets: List of Modal secret names to mount.

    Example:
        import modal
        cfg = FunctionConfig(
            image=modal.Image.debian_slim().pip_install("scipy"),
            gpu="A10G",
        )
    """

    image: Any = None  # modal.Image — None means use TRAINING_IMAGE
    gpu: str | None = None
    timeout: int = 300
    secrets: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Result from remote code execution (sandbox or function)."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1


class RewardEnvironment(ABC):
    """Base class for reward environments.

    Users subclass this and override score() to define custom reward logic.
    Two execution tools are available for running code remotely:

    - execute_in_sandbox(): Isolated per-call containers. No GPU.
      Best for untrusted code.
    - execute_in_function(): Pre-warmed containers. GPU support.
      Best for trusted computation.

    Both accept a modal.Image directly — no wrapper needed.

    Attributes:
        name: Human-readable name for logging.
        sandbox_config: Config for execute_in_sandbox().
        function_config: Config for execute_in_function().
        weight: Weight when combining multiple rewards.
    """

    name: str = "base"
    sandbox_config: SandboxConfig = None
    function_config: FunctionConfig = None
    weight: float = 1.0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.sandbox_config is None:
            cls.sandbox_config = SandboxConfig()
        if cls.function_config is None:
            cls.function_config = FunctionConfig()

    def __init__(self):
        if self.sandbox_config is None:
            self.sandbox_config = SandboxConfig()
        if self.function_config is None:
            self.function_config = FunctionConfig()

    @abstractmethod
    def score(self, completion: str, prompt: str, **kwargs) -> float:
        """Score a single completion.

        Args:
            completion: The model's generated text.
            prompt: The input prompt.
            **kwargs: Additional dataset columns (e.g. testcases, ground_truth).

        Returns:
            Float reward score. Convention: 0.0 = worst, 1.0 = best.
        """
        ...

    def score_batch(
        self, completions: list[str], prompts: list[str], **kwargs
    ) -> list[float]:
        """Score a batch of completions. Override for parallel execution.

        Default implementation loops score() per item, unpacking kwargs.
        """
        results = []
        for i in range(len(completions)):
            item_kw = {k: v[i] for k, v in kwargs.items()}
            results.append(self.score(completions[i], prompts[i], **item_kw))
        return results

    def setup(self) -> None:
        """Called once before the first scoring call. Override for initialization."""
        pass

    def teardown(self) -> None:
        """Called after training ends. Override for cleanup."""
        pass

    # --- Sandbox execution (isolated, no GPU) ---

    def execute_in_sandbox(self, code: str) -> ExecutionResult:
        """Execute code in a Modal Sandbox.

        Creates a fresh isolated container per call using sandbox_config.image.
        Best for untrusted code.
        """
        from mortal.rewards.sandbox_executor import execute_in_sandbox_remote

        return execute_in_sandbox_remote(code, self.sandbox_config)

    def execute_batch_in_sandbox(self, codes: list[str]) -> list[ExecutionResult]:
        """Execute multiple code strings in parallel Modal Sandboxes."""
        from mortal.rewards.sandbox_executor import execute_batch_in_sandbox_remote

        return execute_batch_in_sandbox_remote(codes, self.sandbox_config)

    # --- Function execution (pre-warmed, GPU support) ---

    def execute_in_function(self, code: str) -> ExecutionResult:
        """Execute code in a Modal Function.

        Uses pre-warmed containers from function_config.image.
        Supports GPU. Best for trusted computation.
        """
        from mortal.rewards.function_executor import execute_in_function_remote

        return execute_in_function_remote(code, self.function_config)

    def execute_batch_in_function(self, codes: list[str]) -> list[ExecutionResult]:
        """Execute multiple code strings in parallel Modal Functions."""
        from mortal.rewards.function_executor import execute_batch_in_function_remote

        return execute_batch_in_function_remote(codes, self.function_config)
