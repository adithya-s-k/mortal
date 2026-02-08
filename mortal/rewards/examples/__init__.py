"""Example reward environments.

These are ready-to-use implementations of RewardEnvironment for common
patterns. Use them directly or as reference for building your own.

Available:
    CodeExecutionEnvironment - Execute code in Modal Sandboxes, score by test pass/fail
    LLMJudgeEnvironment      - Use an LLM (OpenAI-compatible API) to score completions
"""

from mortal.rewards.examples.code_execution import CodeExecutionEnvironment
from mortal.rewards.examples.llm_judge import LLMJudgeEnvironment

__all__ = [
    "CodeExecutionEnvironment",
    "LLMJudgeEnvironment",
]
