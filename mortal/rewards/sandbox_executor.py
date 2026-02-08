"""Modal Sandbox execution for reward environments.

Creates sandboxes directly using modal.Sandbox.create(image=...).
No spec-dict layer — takes the modal.Image from SandboxConfig as-is.
"""

from concurrent.futures import ThreadPoolExecutor

import modal

from mortal.app import app
from mortal.rewards.base import ExecutionResult, SandboxConfig


def _run_one_sandbox(code: str, config: SandboxConfig) -> ExecutionResult:
    """Create a sandbox, run code, return result."""
    image = config.image if config.image is not None else modal.Image.debian_slim()
    secrets = [modal.Secret.from_name(n) for n in config.secrets] if config.secrets else []

    sb = None
    try:
        sb = modal.Sandbox.create(app=app, image=image, secrets=secrets)
        p = sb.exec("python", "-c", code, timeout=config.timeout)
        p.wait()
        return ExecutionResult(
            success=p.returncode == 0,
            stdout=p.stdout.read(),
            stderr=p.stderr.read(),
            returncode=p.returncode,
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=str(e),
            returncode=-1,
        )
    finally:
        if sb is not None:
            sb.terminate()


def execute_in_sandbox_remote(code: str, sandbox_config: SandboxConfig) -> ExecutionResult:
    """Execute code in a Modal Sandbox.

    Args:
        code: Python code string to execute.
        sandbox_config: Sandbox configuration with modal.Image.

    Returns:
        ExecutionResult with execution details.
    """
    return _run_one_sandbox(code, sandbox_config)


def execute_batch_in_sandbox_remote(
    codes: list[str], sandbox_config: SandboxConfig
) -> list[ExecutionResult]:
    """Execute multiple code strings in parallel Modal Sandboxes.

    Each sandbox is created independently. Uses threads since each
    Sandbox.create() is an API call to Modal's infrastructure.

    Args:
        codes: List of Python code strings.
        sandbox_config: Sandbox configuration (shared across all).

    Returns:
        List of ExecutionResult, one per code string.
    """
    max_workers = min(len(codes), 16)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda c: _run_one_sandbox(c, sandbox_config), codes))
