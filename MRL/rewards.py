"""Reward dispatch layer for pluggable reward functions."""


def compute_rewards(reward_funcs, completions, **kwargs) -> list[float]:
    """Compute rewards using the configured reward function(s).

    Args:
        reward_funcs: One of:
            - None or "sandbox": delegates to existing sandbox reward worker
            - callable: runs directly with (completions=..., **kwargs)
            - list of callables: runs each, averages results
        completions: List of model completions
        **kwargs: Additional keyword arguments passed to reward functions
            (e.g. testcases, ground_truths, etc.)

    Returns:
        List of float rewards, one per completion.
    """
    if reward_funcs is None or reward_funcs == "sandbox":
        from MRL.workers.reward import reward_helper_function

        return list(
            reward_helper_function(completions, kwargs.get("testcases", []))
        )

    if callable(reward_funcs):
        reward_funcs = [reward_funcs]

    all_rewards = [func(completions=completions, **kwargs) for func in reward_funcs]

    if len(all_rewards) == 1:
        return list(all_rewards[0])

    # Average across reward functions
    return [
        sum(r[i] for r in all_rewards) / len(all_rewards)
        for i in range(len(completions))
    ]
