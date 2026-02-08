"""Worker modules for MORTAL training.

Workers:
- ActorWorker: Training worker using TRL's GRPOTrainer (runs on H100)
- RolloutWorker: vLLM inference worker for fast generation (runs on A10G)
- RewardWorker: Reward computation using Modal Sandboxes for code execution
"""

from mortal.workers.actor import ActorWorker
from mortal.workers.rollout import RolloutWorker
from mortal.workers.reward import (
    compute_reward,
    reward_helper_function,
    compute_reward_batch,
    compute_reward_with_partial_credit,
    partial_credit_reward_function,
)

__all__ = [
    "ActorWorker",
    "RolloutWorker",
    "compute_reward",
    "reward_helper_function",
    "compute_reward_batch",
    "compute_reward_with_partial_credit",
    "partial_credit_reward_function",
]
