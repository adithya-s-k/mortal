"""LLM-as-judge reward environment."""

import os
import re
from concurrent.futures import ThreadPoolExecutor

from mortal.rewards.base import RewardEnvironment, SandboxConfig

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert judge evaluating the quality of AI-generated responses. "
    "Provide only a numeric score."
)

DEFAULT_JUDGING_TEMPLATE = (
    "Rate the following response on a scale of {score_min} to {score_max}.\n\n"
    "Prompt: {prompt}\n\n"
    "Response: {completion}\n\n"
    "Provide ONLY a numeric score between {score_min} and {score_max}."
)


class LLMJudgeEnvironment(RewardEnvironment):
    """Reward environment that uses an LLM as judge.

    Makes API calls to an OpenAI-compatible endpoint to score completions.
    Runs locally (no sandbox needed) - the API calls happen from the
    orchestrator process.

    Examples:
        judge = LLMJudgeEnvironment(
            model="gpt-4o-mini",
            system_prompt="You are a code quality expert.",
            judging_template=(
                "Rate this solution 0-10.\\n\\n"
                "Problem: {prompt}\\nSolution: {completion}"
            ),
            score_range=(0, 10),
        )
        score = judge.score("def add(a,b): return a+b", "Write an add function")
    """

    name = "LLMJudge"
    sandbox_config = SandboxConfig()  # Not used, but required by base class

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        judging_template: str = DEFAULT_JUDGING_TEMPLATE,
        score_range: tuple[int, int] = (0, 10),
        normalize: bool = True,
        api_key_env_var: str | None = None,
        api_base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
        max_retries: int = 2,
        max_workers: int = 8,
        weight: float = 1.0,
    ):
        """Initialize LLMJudgeEnvironment.

        Args:
            model: Model name for the judge (e.g. "gpt-4o-mini").
            provider: API provider. Currently "openai" (OpenAI-compatible).
            system_prompt: System prompt for the judge.
            judging_template: Template string with {prompt}, {completion},
                {score_min}, {score_max}, and any dataset column placeholders.
            score_range: (min, max) score range for the judge output.
            normalize: If True, normalize scores to [0, 1].
            api_key_env_var: Environment variable name for the API key.
                Defaults to OPENAI_API_KEY.
            api_base_url: Custom API base URL (for compatible endpoints).
            temperature: Sampling temperature for the judge.
            max_tokens: Maximum tokens in judge response.
            max_retries: Number of retries on API failure.
            max_workers: Max concurrent API calls for batch scoring.
            weight: Weight when combining multiple reward environments.
        """
        self.model = model
        self.provider = provider
        self.system_prompt = system_prompt
        self.judging_template = judging_template
        self.score_range = score_range
        self.normalize = normalize
        self.api_key_env_var = api_key_env_var or "OPENAI_API_KEY"
        self.api_base_url = api_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.weight = weight
        self._client = None

    def setup(self) -> None:
        """Initialize the OpenAI-compatible client."""
        self._ensure_client()

    def _ensure_client(self):
        """Lazily initialize the API client."""
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "LLMJudgeEnvironment requires the 'openai' package. "
                "Install it with: pip install openai"
            )
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {self.api_key_env_var} "
                f"environment variable."
            )
        kwargs = {"api_key": api_key}
        if self.api_base_url:
            kwargs["base_url"] = self.api_base_url
        self._client = OpenAI(**kwargs)

    def score(self, completion: str, prompt: str, **kwargs) -> float:
        """Score a single completion using the LLM judge.

        Args:
            completion: Model completion to judge.
            prompt: The input prompt.
            **kwargs: Additional dataset columns available in the template.

        Returns:
            Float score, normalized to [0, 1] if normalize=True.
        """
        self._ensure_client()

        score_min, score_max = self.score_range
        user_msg = self.judging_template.format(
            prompt=prompt,
            completion=completion,
            score_min=score_min,
            score_max=score_max,
            **kwargs,
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = response.choices[0].message.content
                return self._parse_score(text)
            except Exception as e:
                if attempt == self.max_retries:
                    print(f"LLMJudge scoring failed after {self.max_retries + 1} attempts: {e}")
                    return 0.0

        return 0.0

    def _parse_score(self, text: str) -> float:
        """Extract a numeric score from the judge's response.

        Finds the first number in the text, clamps to score_range,
        and optionally normalizes to [0, 1].
        """
        score_min, score_max = self.score_range

        # Find all numbers (int or float) in the text
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        if not numbers:
            return 0.0

        raw_score = float(numbers[0])

        # Clamp to range
        raw_score = max(score_min, min(score_max, raw_score))

        if self.normalize and score_max != score_min:
            return (raw_score - score_min) / (score_max - score_min)
        return raw_score

    def score_batch(
        self, completions: list[str], prompts: list[str], **kwargs
    ) -> list[float]:
        """Score a batch using concurrent API calls."""
        self._ensure_client()

        def _score_one(i: int) -> float:
            item_kw = {k: v[i] for k, v in kwargs.items()}
            return self.score(completions[i], prompts[i], **item_kw)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(_score_one, range(len(completions))))

        return results

    def teardown(self) -> None:
        """Clean up the API client."""
        self._client = None
