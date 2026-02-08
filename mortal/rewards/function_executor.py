"""Modal Function execution for reward environments.

Functions use the pre-defined TRAINING_IMAGE — pre-warmed, fast, has
torch/trl/numpy/transformers. For custom images, use sandboxes instead.
"""

from mortal.app import app, TRAINING_IMAGE
from mortal.rewards.base import ExecutionResult, FunctionConfig


def _exec_code(code: str) -> dict:
    """Execute a code string via exec(), capturing stdout/stderr."""
    import io
    import sys
    import traceback as tb

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_out = io.StringIO()
    sys.stderr = captured_err = io.StringIO()

    try:
        exec(code, {"__builtins__": __builtins__})
        return {
            "success": True,
            "stdout": captured_out.getvalue(),
            "stderr": captured_err.getvalue(),
            "returncode": 0,
        }
    except Exception:
        return {
            "success": False,
            "stdout": captured_out.getvalue(),
            "stderr": captured_err.getvalue() + tb.format_exc(),
            "returncode": 1,
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@app.function(image=TRAINING_IMAGE, timeout=600)
def _run_on_training_image(code: str) -> dict:
    return _exec_code(code)


def execute_in_function_remote(
    code: str, function_config: FunctionConfig
) -> ExecutionResult:
    """Execute code in a pre-warmed Modal Function (TRAINING_IMAGE).

    Uses the built-in TRAINING_IMAGE which has torch, trl, numpy,
    transformers, etc. pre-installed. For custom images, use
    execute_in_sandbox() instead.
    """
    result = _run_on_training_image.remote(code)
    return ExecutionResult(
        success=result["success"],
        stdout=result["stdout"],
        stderr=result["stderr"],
        returncode=result["returncode"],
    )


def execute_batch_in_function_remote(
    codes: list[str], function_config: FunctionConfig
) -> list[ExecutionResult]:
    """Execute multiple code strings in parallel via starmap."""
    args = [(code,) for code in codes]
    raw_results = list(_run_on_training_image.starmap(args))
    return [
        ExecutionResult(
            success=r["success"],
            stdout=r["stdout"],
            stderr=r["stderr"],
            returncode=r["returncode"],
        )
        for r in raw_results
    ]
