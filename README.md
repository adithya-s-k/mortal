# MRL - Modal Reinforcement Learning

A serverless reinforcement learning framework for LLMs on [Modal](https://modal.com). Built on TRL and vLLM, MRL supports any RL algorithm available in TRL (GRPO, PPO, DPO, etc.) with pluggable reward environments, configurable GPU allocation, and a simple programmatic API. Currently ships with GRPO as the default training algorithm.

## Architecture

```
User's Machine (CPU)                    Modal Cloud
+---------------------+      +------------------------------+
|                     |      |                              |
|  MRLTrainer         |      |  ActorWorker (GPU)           |
|  +- config          |----->|  +- RL training step         |
|  +- reward_funcs    |      |  +- weight sync              |
|  +- train()         |      |  +- checkpoint               |
|       |             |      |                              |
|  Orchestrator       |      |  RolloutWorker(s) (GPU)      |
|  (local or Modal)   |----->|  +- vLLM generation          |
|       |             |      |  +- logprob computation      |
|                     |      |                              |
|  RewardEnvironment  |      |  Sandbox / Function          |
|  (local scoring     |----->|  +- code execution           |
|   + remote exec)    |      |  +- custom images            |
|                     |      |                              |
+---------------------+      +------------------------------+
```

- **ActorWorker**: RL training step on GPU (A100 by default, configurable)
- **RolloutWorker**: vLLM inference on GPU (A10G by default, configurable)
- **Reward**: Custom `RewardEnvironment`, plain callables, or built-in sandbox mode

**Two orchestration modes:**
- `reward_funcs=None/"sandbox"` -- orchestrator + rewards run on Modal
- `reward_funcs=callable/RewardEnvironment` -- orchestrator runs locally, GPU work on Modal

## Quick Start

### Prerequisites

```bash
pip install modal
modal setup  # authenticate
modal volume create grpo-trl-storage  # one-time
```

### Usage

#### Sandbox Rewards (code execution)

```python
from MRL import MRLTrainer

trainer = MRLTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs="sandbox",  # or None
    max_steps=5,
    actor_gpu="A100",
)
trainer.train()
```

#### Custom Reward Function

```python
from MRL import MRLTrainer
from datasets import load_dataset

dataset = load_dataset("my_dataset", split="train")
# dataset must have a "prompt" column
# all other columns are passed as kwargs to the reward function

def accuracy_reward(completions, ground_truths, **kwargs):
    return [1.0 if c.strip() == gt.strip() else 0.0
            for c, gt in zip(completions, ground_truths)]

trainer = MRLTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
    num_generations=4,
    max_steps=10,
)
trainer.train()
```

#### Multiple Reward Functions with Weights

```python
def format_reward(completions, **kwargs):
    return [1.0 if "```" in c else 0.0 for c in completions]

def length_reward(completions, **kwargs):
    return [min(len(c) / 500, 1.0) for c in completions]

trainer = MRLTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[format_reward, length_reward],
    reward_weights=[0.7, 0.3],  # optional, defaults to equal weights
    train_dataset=dataset,
)
trainer.train()
```

## Reward System

MRL provides a pluggable reward system built around the `RewardEnvironment` base class. You can use plain callables for simple cases, or subclass `RewardEnvironment` for structured reward logic with access to remote execution tools.

### Reward Modes

| Mode | `reward_funcs=` | Where it runs | Use case |
|------|-----------------|---------------|----------|
| Sandbox | `None` or `"sandbox"` | Modal Sandbox | Code execution with test cases |
| Custom callable | `callable` | Local CPU | Any simple reward logic |
| RewardEnvironment | `RewardEnvironment` subclass | Local + remote exec | Structured rewards with sandbox/function access |
| Mixed | `[env, callable, ...]` | Mixed | Multiple rewards, weighted average |

### RewardEnvironment Base Class

Subclass `RewardEnvironment` to create custom reward environments with a fixed input/output contract and access to two remote execution tools:

- **`execute_in_sandbox(code)`** -- Runs code in an isolated Modal Sandbox. Custom `modal.Image` support. Best for untrusted code.
- **`execute_in_function(code)`** -- Runs code in a pre-warmed Modal Function (TRAINING_IMAGE with torch/trl/numpy). Faster, no container spin-up.

```python
from MRL.rewards import RewardEnvironment

class XMLFormatReward(RewardEnvironment):
    name = "XMLFormat"

    def score(self, completion: str, prompt: str, **kwargs) -> float:
        has_think = "<think>" in completion and "</think>" in completion
        has_answer = "<answer>" in completion and "</answer>" in completion
        return float(has_think) * 0.5 + float(has_answer) * 0.5

trainer = MRLTrainer(
    reward_funcs=XMLFormatReward(),
    train_dataset=dataset,
)
```

#### Using Sandbox Execution (custom image)

```python
import modal
from MRL.rewards import RewardEnvironment, SandboxConfig

class NumpyTestEnvironment(RewardEnvironment):
    name = "NumpyTest"
    sandbox_config = SandboxConfig(
        image=modal.Image.debian_slim().pip_install("numpy", "scipy"),
        timeout=60,
    )

    def score(self, completion: str, prompt: str, **kwargs) -> float:
        code = self._extract(completion)
        tests = kwargs.get("testcases", [])
        full_code = f"{code}\n\n" + "\n".join(tests)
        result = self.execute_in_sandbox(full_code)
        return 1.0 if result.success else 0.0
```

#### Using Function Execution (pre-warmed, fast)

```python
from MRL.rewards import RewardEnvironment

class FastCodeCheck(RewardEnvironment):
    name = "FastCodeCheck"

    def score(self, completion: str, prompt: str, **kwargs) -> float:
        # Runs on TRAINING_IMAGE (has torch, numpy, transformers)
        result = self.execute_in_function(f"import torch; print(torch.__version__)")
        return 1.0 if result.success else 0.0
```

#### Batch Scoring

Override `score_batch()` for parallel execution:

```python
def score_batch(self, completions, prompts, **kwargs):
    codes = [self.build_code(c, kwargs["testcases"][i]) for i, c in enumerate(completions)]
    results = self.execute_batch_in_sandbox(codes)  # parallel sandboxes
    return [1.0 if r.success else 0.0 for r in results]
```

### Example Environments

MRL ships with ready-to-use environments in `MRL.rewards.examples`:

#### CodeExecutionEnvironment

Extracts code from completions, runs with test cases in Modal Sandboxes.

```python
from MRL.rewards.examples import CodeExecutionEnvironment
from MRL.rewards import SandboxConfig
import modal

# Default (binary pass/fail)
env = CodeExecutionEnvironment()

# With partial credit and custom image
env = CodeExecutionEnvironment(
    partial_credit=True,
    sandbox_cfg=SandboxConfig(
        image=modal.Image.debian_slim().pip_install("numpy", "pandas"),
        timeout=60,
    ),
)

trainer = MRLTrainer(reward_funcs=env, train_dataset=ds)
```

#### LLMJudgeEnvironment

Uses an OpenAI-compatible API to score completions with an LLM.

```python
from MRL.rewards.examples import LLMJudgeEnvironment

judge = LLMJudgeEnvironment(
    model="gpt-4o-mini",
    system_prompt="You are a code quality expert.",
    judging_template="Rate this solution 0-10.\n\nProblem: {prompt}\nSolution: {completion}",
    score_range=(0, 10),
)

trainer = MRLTrainer(reward_funcs=judge, train_dataset=ds)
```

### Plain Callable (simplest)

```python
def my_reward(completions, ground_truths, metadata, **kwargs):
    # completions: list[str] - model outputs
    # ground_truths, metadata: from dataset columns
    return [float_score_per_completion]
```

## Configuration

### MRLTrainer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model name |
| `reward_funcs` | `None` | Reward function(s), RewardEnvironment(s), or `"sandbox"` |
| `reward_weights` | `None` | Weights for combining multiple rewards |
| `train_dataset` | `None` | HuggingFace Dataset (required for custom rewards) |
| `actor_gpu` | `A100` | GPU for training worker |
| `rollout_gpu` | `A10G` | GPU for vLLM rollout workers |
| `num_rollout_workers` | `2` | Number of parallel rollout workers |
| `num_generations` | `4` | Generations per prompt |
| `batch_size` | `8` | Prompts per training step |
| `learning_rate` | `5e-6` | Learning rate |
| `max_steps` | `-1` | Max training steps (-1 = use epochs) |
| `num_epochs` | `5` | Number of training epochs |
| `loss_type` | `dapo` | Loss variant (grpo/dapo/dr_grpo/bnpo/cispo/sapo) |
| `weight_sync_method` | `reload` | How to sync weights (reload/volume/direct/checkpoint) |
| `use_lora` | `False` | Enable LoRA training |
| `beta` | `0.0` | KL penalty coefficient |
| `epsilon` | `0.2` | PPO-style clipping epsilon |
| `scale_rewards` | `group` | Reward scaling (group/batch/none) |
| `max_tokens` | `8000` | Max tokens per generated completion |
| `max_model_len` | `16384` | Max model context length |
| `max_completion_length` | `1024` | Max completion length for training |

### Weight Sync Methods

| Method | Speed | Description |
|--------|-------|-------------|
| `reload` (default) | Fast (~2-3s) | vLLM v1 sleep/wake_up/reload_weights pattern |
| `volume` | Medium (~10-20s) | Save to shared volume, workers reload |
| `direct` | Fast | In-memory transfer via vLLM load_weights (WIP) |
| `checkpoint` | Slow (~30-40s) | Full checkpoint save + model recreation |

## How It Works

### Training Loop

```
+---------------------------------------------------------------------+
|                        TRAINING LOOP                                |
+---------------------------------------------------------------------+
|                                                                     |
|  1. GET BATCH                                                       |
|     Orchestrator fetches batch of prompts from dataset              |
|                          |                                          |
|                          v                                          |
|  2. GENERATE COMPLETIONS (Parallel)                                 |
|     Fan out across RolloutWorkers (vLLM inference + logprobs)       |
|     +-------------+ +-------------+                                 |
|     |RolloutWorker| |RolloutWorker|                                 |
|     +------+------+ +------+------+                                 |
|            +-------+-------+                                        |
|                    v                                                |
|  3. COMPUTE REWARDS                                                 |
|     RewardEnvironment.score_batch() or callable                     |
|     Can use sandbox/function execution for remote code runs         |
|                    |                                                |
|                    v                                                |
|  4. TRAIN STEP                                                      |
|     ActorWorker: RL loss computation + weight update                |
|                    |                                                |
|                    v                                                |
|  5. SYNC WEIGHTS (Every N steps)                                    |
|     Actor -> RolloutWorkers via reload/volume/checkpoint            |
|                    |                                                |
|                    v                                                |
|  6. REPEAT until max_steps reached                                  |
|                                                                     |
+---------------------------------------------------------------------+
```

### Default Algorithm: GRPO

GRPO (Group Relative Policy Optimization) generates multiple completions per prompt and computes relative advantages within each group:

1. Generate `num_generations` completions per prompt
2. Score each with the reward function
3. Normalize rewards within each group (zero-mean, unit-variance)
4. Update policy to increase probability of higher-reward completions

Supported loss variants: `dapo`, `grpo`, `dr_grpo`, `bnpo`, `cispo`, `sapo`

## File Structure

```
MRL/
+-- __init__.py          # Package exports
+-- app.py               # Modal app, images, volume
+-- config.py            # OrchestratorConfig, ModelConfig, TrainingConfig, GenerationConfig
+-- orchestrator.py      # train() [Modal], train_local() [local], training loop
+-- trainer.py           # MRLTrainer (programmatic API)
+-- rewards/
|   +-- __init__.py      # Core exports (RewardEnvironment, SandboxConfig, etc.)
|   +-- base.py          # RewardEnvironment, SandboxConfig, FunctionConfig, ExecutionResult
|   +-- dispatch.py      # compute_rewards() dispatch layer
|   +-- sandbox_executor.py  # Modal Sandbox execution
|   +-- function_executor.py # Modal Function execution (TRAINING_IMAGE)
|   +-- utils.py         # Code extraction helpers
|   +-- examples/
|       +-- code_execution.py  # CodeExecutionEnvironment
|       +-- llm_judge.py       # LLMJudgeEnvironment
+-- workers/
    +-- actor.py         # ActorWorker - RL training, weight management
    +-- rollout.py       # RolloutWorker - vLLM generation, weight sync
```

## Testing

```bash
# Unit tests (reward environments only, no GPU)
python tests/test_trainer.py --unit-only

# Training tests (full loop with GPU workers)
python tests/test_trainer.py --train-only

# All tests
python tests/test_trainer.py
```

## Volume Management

```bash
# List checkpoints
modal volume ls grpo-trl-storage checkpoints/

# Clear model cache (if corrupted)
modal volume rm grpo-trl-storage model_cache/Qwen_Qwen2-0.5B-Instruct
```

## Monitoring

- **Modal Dashboard**: https://modal.com/apps -- logs, costs, GPU usage
- **Weights & Biases**: metrics logged to project `modal-grpo-trl`

## Troubleshooting

### Out of Memory
- Reduce `batch_size`, `max_model_len`, or `num_generations`
- Use a smaller model or enable LoRA (`use_lora=True`)

### Slow Generation
- Increase `num_rollout_workers` for more parallelism
- Reduce `max_tokens` for shorter completions

### Weight Sync Issues
- Default `reload` method requires vLLM v1 (>= 0.8.0)
- Fall back to `volume` or `checkpoint` if reload fails

### Function Not Hydrated
- If using `execute_in_function()` from a custom script, import `MRL.rewards.function_executor` before `app.run()` so Modal discovers the function.

## References

- [GRPO Paper (DeepSeek)](https://arxiv.org/abs/2402.03300)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [vLLM Documentation](https://docs.vllm.ai)
- [Modal Documentation](https://modal.com/docs)
- [veRL](https://github.com/volcengine/verl) - Architecture inspiration
