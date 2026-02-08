"""Reward dispatch layer for pluggable reward functions."""

from mortal.rewards.base import RewardEnvironment


def compute_rewards(
    reward_funcs, completions: list[str], prompts: list[str] | None = None,
    weights: list[float] | None = None, **kwargs
) -> list[float]:
    """Compute rewards using the configured reward function(s).

    Handles all reward types: RewardEnvironment instances, plain callables,
    lists of mixed types, and the legacy None/"sandbox" path.

    Args:
        reward_funcs: One of:
            - None or "sandbox": delegates to default CodeExecutionEnvironment
            - RewardEnvironment instance: calls score_batch()
            - callable: runs with (completions=..., **kwargs)
            - list of the above: runs each, combines with weighted average
        completions: List of model completions.
        prompts: List of prompts (one per completion). Defaults to empty strings.
        weights: Optional list of weights for combining multiple reward functions.
            If None, uses RewardEnvironment.weight for environments and 1.0 for callables.
        **kwargs: Additional keyword arguments passed to reward functions
            (e.g. testcases, ground_truths, etc.)

    Returns:
        List of float rewards, one per completion.
    """
    prompts = prompts or [""] * len(completions)

    # Legacy: None/"sandbox" -> default CodeExecutionEnvironment
    if reward_funcs is None or reward_funcs == "sandbox":
        from mortal.rewards.examples.code_execution import CodeExecutionEnvironment

        env = CodeExecutionEnvironment()
        return env.score_batch(completions, prompts, **kwargs)

    # Normalize to list
    if not isinstance(reward_funcs, list):
        reward_funcs = [reward_funcs]

    all_rewards = []
    effective_weights = []

    for i, rf in enumerate(reward_funcs):
        if isinstance(rf, RewardEnvironment):
            rewards = rf.score_batch(completions, prompts, **kwargs)
            effective_weights.append(weights[i] if weights else rf.weight)
        elif callable(rf):
            rewards = rf(completions=completions, **kwargs)
            effective_weights.append(weights[i] if weights else 1.0)
        else:
            raise TypeError(
                f"reward_funcs[{i}] must be a RewardEnvironment, callable, or "
                f"'sandbox'/None. Got {type(rf)}"
            )
        all_rewards.append(list(rewards))

    if len(all_rewards) == 1:
        return all_rewards[0]

    # Weighted average across reward functions
    total_weight = sum(effective_weights)
    return [
        sum(
            all_rewards[j][i] * effective_weights[j]
            for j in range(len(all_rewards))
        )
        / total_weight
        for i in range(len(completions))
    ]
