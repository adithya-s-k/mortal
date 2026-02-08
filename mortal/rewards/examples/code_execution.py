"""Code execution reward environment using Modal Sandboxes."""

from typing import Callable

from mortal.rewards.base import ExecutionResult, RewardEnvironment, SandboxConfig
from mortal.rewards.utils import extract_code_from_completion


class CodeExecutionEnvironment(RewardEnvironment):
    """Reward environment that executes code in Modal Sandboxes.

    Extracts code from model completions, combines with test cases,
    and runs in isolated sandboxes. Supports binary (pass/fail) and
    partial credit scoring.

    Examples:
        # Default: binary scoring
        env = CodeExecutionEnvironment()

        # With partial credit and custom image
        import modal
        env = CodeExecutionEnvironment(
            partial_credit=True,
            sandbox_cfg=SandboxConfig(
                image=modal.Image.debian_slim().pip_install("numpy", "pandas"),
                timeout=60,
            ),
        )

        # Custom code extraction
        env = CodeExecutionEnvironment(
            code_extraction_fn=lambda c: c.split("# Solution")[1],
        )
    """

    name = "CodeExecution"

    def __init__(
        self,
        partial_credit: bool = False,
        sandbox_cfg: SandboxConfig | None = None,
        code_extraction_fn: Callable[[str], str] | None = None,
        test_key: str = "testcases",
        weight: float = 1.0,
    ):
        """Initialize CodeExecutionEnvironment.

        Args:
            partial_credit: If True, score = fraction of passing tests.
                If False, score = 1.0 only if all tests pass.
            sandbox_cfg: Custom sandbox configuration. None uses defaults.
            code_extraction_fn: Custom function to extract code from completions.
                If None, uses default markdown block extraction.
            test_key: Key in kwargs containing test cases. Default "testcases".
            weight: Weight when combining multiple reward environments.
        """
        self.partial_credit = partial_credit
        self.sandbox_config = sandbox_cfg or SandboxConfig()
        self._code_extraction_fn = code_extraction_fn
        self.test_key = test_key
        self.weight = weight

    def extract_code(self, completion: str) -> str:
        """Extract code from a completion. Override for custom extraction."""
        if self._code_extraction_fn:
            return self._code_extraction_fn(completion)
        return extract_code_from_completion(completion)

    def build_test_code(self, code: str, testcases: list[str]) -> str:
        """Combine extracted code with test cases. Override for custom assembly."""
        return f"{code}\n\n" + "\n".join(testcases)

    def score(self, completion: str, prompt: str, **kwargs) -> float:
        """Score a single completion by executing code with test cases.

        Args:
            completion: Model completion containing code.
            prompt: The input prompt (unused but part of interface).
            **kwargs: Must contain self.test_key (default "testcases")
                as a list of assert statements.

        Returns:
            1.0 if all tests pass (or fraction if partial_credit), 0.0 otherwise.
        """
        testcases = kwargs.get(self.test_key, [])
        code = self.extract_code(completion)

        if not testcases:
            return 0.0

        if not self.partial_credit:
            full_code = self.build_test_code(code, testcases)
            result = self.execute_in_sandbox(full_code)
            return 1.0 if result.success else 0.0
        else:
            passed = 0
            for t in testcases:
                result = self.execute_in_sandbox(self.build_test_code(code, [t]))
                if result.success:
                    passed += 1
            return passed / len(testcases)

    def score_batch(
        self, completions: list[str], prompts: list[str], **kwargs
    ) -> list[float]:
        """Score a batch using parallel sandbox execution.

        For non-partial-credit mode, all code+tests are submitted in one
        batch via starmap. For partial credit, each test is submitted
        separately.
        """
        all_testcases = kwargs.get(self.test_key, [[] for _ in completions])
        codes = [self.extract_code(c) for c in completions]

        if not self.partial_credit:
            # Build all code strings and execute in parallel
            code_strings = []
            for code, tc in zip(codes, all_testcases):
                if tc:
                    code_strings.append(self.build_test_code(code, tc))
                else:
                    code_strings.append("")

            # Filter out empty ones, track indices
            non_empty = [(i, cs) for i, cs in enumerate(code_strings) if cs]
            if not non_empty:
                return [0.0] * len(completions)

            results_map = {}
            batch_codes = [cs for _, cs in non_empty]
            batch_results = self.execute_batch_in_sandbox(batch_codes)
            for (orig_idx, _), result in zip(non_empty, batch_results):
                results_map[orig_idx] = 1.0 if result.success else 0.0

            return [results_map.get(i, 0.0) for i in range(len(completions))]
        else:
            # Partial credit: fan out per test case
            # Build all (code, single_test) pairs
            tasks = []  # (completion_idx, test_idx, code_string)
            for i, (code, tc_list) in enumerate(zip(codes, all_testcases)):
                if not tc_list:
                    continue
                for j, t in enumerate(tc_list):
                    tasks.append((i, j, self.build_test_code(code, [t])))

            if not tasks:
                return [0.0] * len(completions)

            # Execute all in parallel
            batch_codes = [cs for _, _, cs in tasks]
            batch_results = self.execute_batch_in_sandbox(batch_codes)

            # Aggregate results per completion
            pass_counts: dict[int, int] = {}
            total_counts: dict[int, int] = {}
            for (comp_idx, _, _), result in zip(tasks, batch_results):
                total_counts[comp_idx] = total_counts.get(comp_idx, 0) + 1
                if result.success:
                    pass_counts[comp_idx] = pass_counts.get(comp_idx, 0) + 1

            return [
                pass_counts.get(i, 0) / total_counts[i]
                if i in total_counts and total_counts[i] > 0
                else 0.0
                for i in range(len(completions))
            ]
