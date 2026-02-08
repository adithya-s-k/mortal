"""GSM8K GRPO Training — teach a model to solve math by generating Python code.

Uses MortalTrainer with a custom RewardEnvironment built on mortal's sandbox
execution infrastructure. The model learns to output <reasoning>...</reasoning>
<code>...</code> and the code is executed in isolated Modal Sandboxes.

Usage:
    # Single GPU, no vLLM (simplest)
    python examples/gsm8k_grpo.py --mode single_node

    # Single GPU, vLLM colocate (faster generation)
    python examples/gsm8k_grpo.py --mode single_node_vllm_colocate

    # 2xGPU, vLLM on dedicated GPU
    python examples/gsm8k_grpo.py --mode single_node_vllm_serve

    # Distributed (veRL-style, separate actor + rollout workers)
    python examples/gsm8k_grpo.py --mode distributed

    # With custom settings
    python examples/gsm8k_grpo.py --mode single_node --model Qwen/Qwen2.5-1.5B-Instruct \
        --max_steps 100 --max_samples 500 --batch_size 4 --num_generations 4

    # Fire and forget
    python examples/gsm8k_grpo.py --mode single_node --detach
"""

import argparse
import re

from datasets import load_dataset

from mortal import MortalTrainer, SingleNode, Distributed, GPUConfig
from mortal.rewards.base import RewardEnvironment, FunctionConfig


# ---------------------------------------------------------------------------
# System prompt — instructs the model to generate executable Python code
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful math assistant. Generate Python code that solves the given math problem.
Your code must follow these rules:
1. Use only basic Python operations and built-in functions
2. Always use proper variable assignments
3. Store the final answer in a variable
4. End with a print statement to output the answer
5. Use only numbers and basic math operations (+, -, *, /, //, %, **)

Example format:
<reasoning>
1. First, we'll calculate...
2. Then, we'll...
3. Finally, we'll...
</reasoning>
<code>
# Calculate the result
number1 = 5
number2 = 3
result = number1 + number2

# Print the final answer
print(result)
</code>

Important: Always store calculations in variables and print the final result."""


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def extract_hash_answer(text: str) -> str | None:
    """Extract the numeric answer after #### in GSM8K answers."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def prepare_dataset(max_samples: int | None = None):
    """Load GSM8K and format for GRPO training.

    Returns a dataset with:
    - prompt: list of chat messages (system + user)
    - answer: expected numeric answer (string)
    """
    data = load_dataset("openai/gsm8k", "main", split="train")
    data = data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    return data


# ---------------------------------------------------------------------------
# Helper: extract code from <code> tags
# ---------------------------------------------------------------------------

def extract_xml_code(text: str) -> str | None:
    """Extract code from between <code> tags."""
    if "<code>" not in text or "</code>" not in text:
        return None
    code = text.split("<code>")[-1].split("</code>")[0].strip()
    return code if code else None


# ---------------------------------------------------------------------------
# Custom RewardEnvironment — GSM8K code execution via mortal sandboxes
# ---------------------------------------------------------------------------

class GSM8KCodeExecutionReward(RewardEnvironment):
    """Executes generated code in Modal Functions and checks against expected answer.

    Scoring:
    - 0.0: no code found
    - 0.2: code found with Python-like syntax
    - 0.5: code executes successfully
    - 2.0: code executes AND output matches expected answer

    Uses mortal's execute_batch_in_function() for parallel pre-warmed execution.
    These show up as function calls on the Modal dashboard.
    """

    name = "GSM8KCodeExecution"

    def __init__(self, function_cfg: FunctionConfig | None = None, weight: float = 1.0):
        self.function_config = function_cfg or FunctionConfig(timeout=30)
        self.weight = weight

    def extract_code(self, completion: str) -> str | None:
        """Extract code from <code> tags or ``` markdown blocks."""
        code = extract_xml_code(completion)
        if code is not None:
            return code
        # Fallback: try markdown code blocks
        match = re.search(r"```(?:python)?\s*\n?(.*?)```", completion, re.DOTALL)
        if match:
            return match.group(1).strip() or None
        return None

    def score(self, completion: str, prompt: str, **kwargs) -> float:
        expected = kwargs.get("answer", "")
        code = self.extract_code(completion)
        if not code:
            return 0.0

        base = 0.2 if any(kw in code for kw in ["def", "for", "if", "print", "="]) else 0.0

        # Ensure code prints result
        exec_code = code.strip()
        if "print(" not in exec_code:
            exec_code += "\nprint(result)"

        result = self.execute_in_function(exec_code)
        if not result.success:
            return base

        stdout = result.stdout.strip()
        if not stdout:
            return base + 0.3

        # Check correctness
        if stdout == str(expected).strip():
            return base + 0.3 + 1.5  # 2.0 total
        return base + 0.3  # 0.5 total — executed but wrong answer

    def score_batch(self, completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        """Parallel sandbox execution for all completions."""
        answers = kwargs.get("answer", [""] * len(completions))
        codes = [self.extract_code(c) for c in completions]

        print(f"[GSM8K score_batch] {len(completions)} completions, "
              f"codes found: {sum(1 for c in codes if c is not None)}/{len(codes)}, "
              f"answers: {answers[:2]}...")

        # Build code strings for execution
        exec_codes = []
        has_code = []
        base_rewards = []
        for code in codes:
            if code is None:
                exec_codes.append("")
                has_code.append(False)
                base_rewards.append(0.0)
            else:
                base = 0.2 if any(kw in code for kw in ["def", "for", "if", "print", "="]) else 0.0
                c = code.strip()
                if "print(" not in c:
                    c += "\nprint(result)"
                exec_codes.append(c)
                has_code.append(True)
                base_rewards.append(base)

        # Execute non-empty codes in parallel via mortal's function infrastructure
        non_empty = [(i, c) for i, c in enumerate(exec_codes) if has_code[i]]
        exec_results = {}
        if non_empty:
            batch_codes = [c for _, c in non_empty]
            print(f"[GSM8K score_batch] Executing {len(batch_codes)} codes via Modal Function...")
            print(f"[GSM8K score_batch] First code: {batch_codes[0][:200]}")
            batch_results = self.execute_batch_in_function(batch_codes)
            print(f"[GSM8K score_batch] Function results: {[(r.success, r.stdout[:50] if r.stdout else '') for r in batch_results[:3]]}")
            for (idx, _), result in zip(non_empty, batch_results):
                exec_results[idx] = result
        else:
            print(f"[GSM8K score_batch] No code to execute, skipping")

        # Score
        rewards = []
        for i in range(len(completions)):
            if not has_code[i]:
                rewards.append(0.0)
                continue

            result = exec_results.get(i)
            if result is None or not result.success:
                rewards.append(base_rewards[i])
                continue

            stdout = result.stdout.strip()
            if not stdout:
                rewards.append(base_rewards[i] + 0.3)
                continue

            if stdout == str(answers[i]).strip():
                rewards.append(base_rewards[i] + 0.3 + 1.5)
            else:
                rewards.append(base_rewards[i] + 0.3)

        return rewards


# ---------------------------------------------------------------------------
# TRL-compatible reward functions (format + quality checks)
# ---------------------------------------------------------------------------

def format_reward(completions, **kwargs) -> list[float]:
    """Check completions for XML structure, with partial credit.

    Scoring:
    - 0.5: full <reasoning>...</reasoning><code>...</code>
    - 0.3: has <code>...</code> but missing reasoning
    - 0.1: has ``` code blocks (markdown style)
    - 0.0: no code structure at all
    """
    responses = [c[0]["content"] for c in completions]

    # Debug: print first 2 completions per batch
    for i, r in enumerate(responses[:2]):
        print(f"\n{'='*40} Completion {i} {'='*40}")
        print(r)
        print(f"{'='*90}\n")

    rewards = []
    for r in responses:
        if re.search(r"<reasoning>.*?</reasoning>\s*<code>.*?</code>", r, re.DOTALL):
            rewards.append(0.5)
        elif re.search(r"<code>.*?</code>", r, re.DOTALL):
            rewards.append(0.3)
        elif "```" in r:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def code_quality_reward(completions, **kwargs) -> list[float]:
    """Check code quality — works with <code> tags or ``` markdown blocks."""
    responses = [c[0]["content"] for c in completions]
    rewards = []

    for response in responses:
        # Try <code> tags first, then ``` blocks
        code = extract_xml_code(response)
        if code is None and "```" in response:
            # Extract from markdown code block
            match = re.search(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
            if match:
                code = match.group(1).strip()
        if not code:
            rewards.append(0.0)
            continue
        reward = 0.0
        if "#" in code:
            reward += 0.1
        if re.search(r"print\s*\(", code):
            reward += 0.1
        if re.search(r"^\s{4}|\t", code, re.MULTILINE):
            reward += 0.1
        if any(kw in code for kw in ["=", "for", "if", "def"]):
            reward += 0.1
        rewards.append(reward)

    return rewards


# ---------------------------------------------------------------------------
# Mode builders
# ---------------------------------------------------------------------------

MODES = {
    "single_node": lambda: SingleNode(gpu="A100-80GB"),
    "single_node_vllm_colocate": lambda: SingleNode(
        gpu="A100-80GB", use_vllm=True, vllm_mode="colocate",
    ),
    "single_node_vllm_serve": lambda: SingleNode(
        gpu=GPUConfig("A100-80GB", count=2), use_vllm=True, vllm_mode="serve",
    ),
    "distributed": lambda: Distributed(
        actor="A100", rollout="A10G", num_rollout_workers=1,
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GSM8K GRPO Training with MORTAL")
    parser.add_argument("--mode", choices=list(MODES.keys()), default="single_node",
                        help="Execution mode")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size (None = full 7.5k)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--detach", action="store_true",
                        help="Fire and forget (training continues after terminal closes)")
    args = parser.parse_args()

    # Prepare dataset
    print(f"Loading GSM8K dataset (max_samples={args.max_samples})...")
    dataset = prepare_dataset(max_samples=args.max_samples)
    print(f"Dataset ready: {len(dataset)} samples")

    # Build mode
    mode = MODES[args.mode]()
    print(f"Mode: {args.mode} → {mode}")

    # Build reward functions:
    # - GSM8KCodeExecutionReward: executes code in Modal Sandboxes, checks answer
    # - format_reward: checks <reasoning>/<code> XML structure
    # - code_quality_reward: checks comments, print statements, etc.
    code_exec_env = GSM8KCodeExecutionReward()

    # Create trainer — same DX as TRL, just add mode=
    trainer = MortalTrainer(
        model=args.model,
        mode=mode,
        reward_funcs=[format_reward, code_exec_env, code_quality_reward],
        train_dataset=dataset,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        max_completion_length=2048,
        loss_type="grpo",
        beta=0.001,
    )

    trainer.train(detach=args.detach)


if __name__ == "__main__":
    main()
