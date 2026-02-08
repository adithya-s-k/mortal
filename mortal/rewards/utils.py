"""Utility functions for reward computation."""


def extract_code_from_completion(completion: str) -> str:
    """Extract code from a model completion.

    Handles markdown code blocks (```python ... ```) and plain text.

    Args:
        completion: Model completion text, potentially with markdown code blocks.

    Returns:
        Extracted code string.
    """
    if "```python" in completion:
        start_idx = completion.find("```python") + len("```python")
        end_idx = completion.find("```", start_idx)
        if end_idx != -1:
            return completion[start_idx:end_idx].strip()
        else:
            return completion[start_idx:].strip()
    elif "```" in completion:
        start_idx = completion.find("```") + len("```")
        # Skip language identifier if present
        newline_idx = completion.find("\n", start_idx)
        if newline_idx != -1 and newline_idx - start_idx < 20:
            start_idx = newline_idx + 1
        end_idx = completion.find("```", start_idx)
        if end_idx != -1:
            return completion[start_idx:end_idx].strip()
        else:
            return completion[start_idx:].strip()
    else:
        return completion.strip()
