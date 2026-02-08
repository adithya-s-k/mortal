# MORTAL

**M**odal **O**rchestrated **R**einforcement **T**raining **A**rchitecture for **L**LMs

A serverless reinforcement learning framework for LLMs on [Modal](https://modal.com). Built on [TRL](https://github.com/huggingface/trl) and [vLLM](https://docs.vllm.ai), MORTAL lets you write reward functions exactly like TRL's `GRPOTrainer`, then turbocharge training with Modal's cloud GPUs. Pluggable reward environments, configurable execution modes, and a simple programmatic API.

---

## Overview

```
                           ┌─────────────────────────────────────────────────────┐
                           │                  Modal Cloud                        │
                           │                                                     │
┌──────────────┐           │  ┌─────────────────────────────────────────────┐    │
│              │           │  │         SingleNode Mode                     │    │
│  Your Code   │           │  │  ┌─────────────┐  ┌──────────────────┐     │    │
│              │           │  │  │ TRL GRPO     │  │ vLLM (optional)  │     │    │
│  from mortal │  ────────>│  │  │ Trainer      │  │ colocate / serve │     │    │
│  import ...  │           │  │  └─────────────┘  └──────────────────┘     │    │
│              │           │  └─────────────────────────────────────────────┘    │
│  trainer =   │           │                       OR                            │
│  MortalTrain │           │  ┌─────────────────────────────────────────────┐    │
│  er(...)     │           │  │       Distributed Mode (veRL-style)         │    │
│              │           │  │                                             │    │
│  trainer     │  ────────>│  │  Orchestrator ──> ActorWorker (A100)        │    │
│  .train()    │           │  │       │           ├─ Train step             │    │
│              │           │  │       │           ├─ Weight sync            │    │
└──────────────┘           │  │       │           └─ Checkpoint             │    │
                           │  │       │                                     │    │
                           │  │       └────────> RolloutWorker(s) (A10G)    │    │
                           │  │                  ├─ vLLM generation         │    │
                           │  │                  └─ Logprob computation     │    │
                           │  └─────────────────────────────────────────────┘    │
                           │                                                     │
                           │  ┌─────────────────────────────────────────────┐    │
                           │  │           Reward Execution                  │    │
                           │  │  ┌──────────┐  ┌───────────┐  ┌─────────┐  │    │
                           │  │  │ Sandbox  │  │ Function  │  │ Custom  │  │    │
                           │  │  │ (isolated│  │ (pre-warm │  │ Callable│  │    │
                           │  │  │  per-call│  │  fast)    │  │         │  │    │
                           │  │  └──────────┘  └───────────┘  └─────────┘  │    │
                           │  └─────────────────────────────────────────────┘    │
                           └─────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

```bash
pip install modal
modal setup  # authenticate
modal volume create grpo-trl-storage  # one-time
```

### Minimal Example

```python
from mortal import MortalTrainer, SingleNode

trainer = MortalTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    mode=SingleNode(gpu="A100"),
    max_steps=5,
)
trainer.train()
```

### GSM8K Example (full end-to-end)

See [`examples/gsm8k_grpo.py`](examples/gsm8k_grpo.py) -- a complete example that teaches a model to solve math problems by generating and executing Python code. Demonstrates custom `RewardEnvironment`, multiple reward functions, and all execution modes.

```bash
# Single GPU, no vLLM (simplest)
python examples/gsm8k_grpo.py --mode single_node

# Single GPU with vLLM colocated (faster generation)
python examples/gsm8k_grpo.py --mode single_node_vllm_colocate

# 2x GPU with vLLM on dedicated GPU
python examples/gsm8k_grpo.py --mode single_node_vllm_serve

# Distributed (separate actor + rollout workers)
python examples/gsm8k_grpo.py --mode distributed

# With custom settings
python examples/gsm8k_grpo.py --mode single_node --model Qwen/Qwen3-0.6B \
    --max_steps 100 --batch_size 4 --num_generations 4
```

---

## Execution Modes

### SingleNode

Everything runs in **one Modal container**. TRL's `GRPOTrainer` handles the training loop natively. Simplest to use, great for small-to-medium models.

```
┌───────────────────────────────────────────────┐
│              Single Container (GPU)            │
│                                               │
│   ┌───────────────┐    ┌──────────────────┐   │
│   │  TRL GRPO     │    │  vLLM (optional) │   │
│   │  Trainer      │◄──►│  generation      │   │
│   │  + training   │    │  + logprobs      │   │
│   └───────┬───────┘    └──────────────────┘   │
│           │                                   │
│           ▼                                   │
│   ┌───────────────┐                           │
│   │  Reward Funcs │ ──► Sandbox / Function    │
│   └───────────────┘                           │
└───────────────────────────────────────────────┘
```

```python
from mortal import MortalTrainer, SingleNode, GPUConfig

# Basic -- no vLLM
trainer = MortalTrainer(mode=SingleNode(gpu="A100"), ...)

# vLLM colocated on same GPU (faster generation)
trainer = MortalTrainer(mode=SingleNode(gpu="A100", use_vllm=True, vllm_mode="colocate"), ...)

# vLLM on separate GPU (requires 2+ GPUs)
trainer = MortalTrainer(
    mode=SingleNode(gpu=GPUConfig("A100", count=2), use_vllm=True, vllm_mode="serve"),
    ...
)
```

### Distributed

veRL-style orchestration with **separate workers** for training and generation. Supports horizontal scaling of rollout workers and heterogeneous GPUs.

```
┌──────────────────────────────────────────────────────────────┐
│                     Orchestrator (Modal)                      │
│                                                              │
│   ┌──────────────────┐       ┌───────────────────────────┐   │
│   │  ActorWorker     │       │  RolloutWorker(s)         │   │
│   │  (A100)          │       │  (A10G x N)               │   │
│   │                  │       │                           │   │
│   │  ├─ Train step   │  ◄──  │  ├─ vLLM generation      │   │
│   │  ├─ Loss compute │  ──►  │  ├─ Logprob computation  │   │
│   │  ├─ Weight sync ─┼──────►│  └─ Weight reload        │   │
│   │  └─ Checkpoint   │       │                           │   │
│   └──────────────────┘       └───────────────────────────┘   │
│             │                                                │
│             ▼                                                │
│   ┌──────────────────┐                                       │
│   │  Reward Layer    │ ──► Sandbox / Function / Callable     │
│   └──────────────────┘                                       │
└──────────────────────────────────────────────────────────────┘
```

```python
from mortal import MortalTrainer, Distributed

trainer = MortalTrainer(
    mode=Distributed(actor="A100", rollout="A10G", num_rollout_workers=2),
    ...
)
```

Both modes support custom reward functions, custom datasets, and `detach=True` for fire-and-forget training.

---

## Training Loop

```
    ┌─────────────────┐
    │  1. GET BATCH    │  Fetch prompts from dataset
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  2. GENERATE     │  Fan out across RolloutWorker(s) with vLLM
    │                  │  (or TRL's built-in generation in SingleNode)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  3. REWARD       │  Score completions via reward_funcs
    │                  │  (sandbox / function / callable / RewardEnvironment)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  4. TRAIN STEP   │  GRPO loss computation + weight update
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  5. SYNC WEIGHTS │  Push updated weights to RolloutWorker(s)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  6. REPEAT       │  Until max_steps reached
    └─────────────────┘
```

**GRPO** (Group Relative Policy Optimization) generates multiple completions per prompt, scores them, normalizes rewards within each group, and updates the policy to favor higher-reward completions.

Supported loss variants via TRL: `grpo`, `dapo`, `dr_grpo`, `bnpo`, `cispo`, `sapo`

---

## Reward System

MORTAL's reward system is designed to feel identical to TRL. Write reward functions the same way, get remote execution for free.

```
┌──────────────────────────────────────────────────────────────┐
│                      Reward Options                          │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │  Plain         │  │  Reward        │  │  Built-in     │  │
│  │  Callable      │  │  Environment   │  │  Environments │  │
│  │                │  │                │  │               │  │
│  │  def reward(   │  │  class MyEnv(  │  │  CodeExec     │  │
│  │    completions │  │    Reward      │  │  Environment  │  │
│  │    **kwargs    │  │    Environment │  │               │  │
│  │  ):            │  │  ):           │  │  LLMJudge     │  │
│  │    return [...] │  │    def score  │  │  Environment  │  │
│  │                │  │    def score  │  │               │  │
│  │                │  │      _batch   │  │               │  │
│  └────────────────┘  └───────┬────────┘  └───────────────┘  │
│                              │                               │
│                    ┌─────────┴──────────┐                    │
│                    │  Remote Execution  │                    │
│                    │                    │                    │
│                    │  execute_in_       │                    │
│                    │  ├─ sandbox()      │                    │
│                    │  └─ function()     │                    │
│                    │                    │                    │
│                    │  execute_batch_in_ │                    │
│                    │  ├─ sandbox()      │                    │
│                    │  └─ function()     │                    │
│                    └───────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

### Plain Callable (simplest)

Works exactly like TRL's reward functions:

```python
def format_reward(completions, **kwargs):
    """completions: list[list[dict]] in TRL chat format."""
    responses = [c[0]["content"] for c in completions]
    return [1.0 if "<answer>" in r else 0.0 for r in responses]

trainer = MortalTrainer(
    reward_funcs=[format_reward],
    train_dataset=dataset,
    mode=SingleNode(gpu="A100"),
)
```

### RewardEnvironment (structured + remote execution)

Subclass `RewardEnvironment` for structured reward logic with access to Modal's execution infrastructure:

```python
from mortal.rewards import RewardEnvironment

class CodeExecutionReward(RewardEnvironment):
    name = "CodeExecution"

    def score(self, completion: str, prompt: str, **kwargs) -> float:
        code = extract_code(completion)
        if not code:
            return 0.0
        result = self.execute_in_function(code)  # runs on Modal
        return 1.0 if result.success else 0.0

    def score_batch(self, completions, prompts, **kwargs):
        codes = [extract_code(c) for c in completions]
        results = self.execute_batch_in_function(codes)  # parallel on Modal
        return [1.0 if r.success else 0.0 for r in results]
```

### Remote Execution Tools

| Method | Container | Speed | Best for |
|--------|-----------|-------|----------|
| `execute_in_sandbox()` | Fresh isolated container | Slower (cold start) | Untrusted code |
| `execute_in_function()` | Pre-warmed TRAINING_IMAGE | Fast (warm container) | Trusted computation |
| `execute_batch_in_sandbox()` | Parallel sandboxes | Parallel | Batch untrusted code |
| `execute_batch_in_function()` | Parallel functions | Parallel + fast | Batch trusted code |

### Multiple Rewards with Weights

```python
trainer = MortalTrainer(
    reward_funcs=[format_reward, CodeExecutionReward(), quality_reward],
    reward_weights=[0.3, 1.0, 0.2],  # optional, defaults to equal
    train_dataset=dataset,
)
```

### Built-in Environments

```python
from mortal.rewards.examples import CodeExecutionEnvironment, LLMJudgeEnvironment

# Code execution with test cases (sandbox-based)
env = CodeExecutionEnvironment(partial_credit=True)

# LLM-as-judge (uses OpenAI-compatible API)
judge = LLMJudgeEnvironment(model="gpt-4o-mini", score_range=(0, 10))

trainer = MortalTrainer(reward_funcs=[env, judge], train_dataset=ds)
```

### Custom Sandbox Image

```python
import modal
from mortal.rewards import RewardEnvironment, SandboxConfig

class NumpyTestEnv(RewardEnvironment):
    name = "NumpyTest"
    sandbox_config = SandboxConfig(
        image=modal.Image.debian_slim().pip_install("numpy", "scipy"),
        timeout=60,
    )

    def score(self, completion: str, prompt: str, **kwargs) -> float:
        result = self.execute_in_sandbox(extract_code(completion))
        return 1.0 if result.success else 0.0
```

---

## Configuration

### MortalTrainer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"Qwen/Qwen2.5-0.5B-Instruct"` | HuggingFace model name |
| `mode` | `Distributed()` | `SingleNode(...)` or `Distributed(...)` |
| `reward_funcs` | `None` | Reward function(s), `RewardEnvironment`(s), or `"sandbox"` |
| `reward_weights` | `None` | Weights for combining multiple rewards |
| `train_dataset` | `None` | HuggingFace Dataset with `prompt` column |
| `num_generations` | `4` | Completions per prompt (GRPO group size) |
| `batch_size` | `8` | Prompts per training step |
| `learning_rate` | `5e-6` | Learning rate |
| `max_steps` | `-1` | Max training steps (`-1` = use epochs) |
| `num_epochs` | `5` | Number of training epochs |
| `loss_type` | `"dapo"` | `grpo` / `dapo` / `dr_grpo` / `bnpo` / `cispo` / `sapo` |
| `beta` | `0.0` | KL penalty coefficient |
| `epsilon` | `0.2` | PPO-style clipping epsilon |
| `scale_rewards` | `"group"` | Reward scaling: `group` / `batch` / `none` |
| `max_completion_length` | `1024` | Max completion length for training |
| `max_tokens` | `8000` | Max tokens per generated completion |
| `max_model_len` | `16384` | Max model context length |
| `use_lora` | `False` | Enable LoRA training |
| `gradient_checkpointing` | `True` | Reduces memory via activation recomputation |

### SingleNode Options

```python
SingleNode(gpu="A100", use_vllm=False, vllm_mode="colocate")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu` | `"A100"` | GPU type (string or `GPUConfig("A100", count=2)`) |
| `use_vllm` | `False` | Use vLLM for faster generation |
| `vllm_mode` | `"colocate"` | `"colocate"` (shared GPU) or `"serve"` (separate GPU, needs 2+) |

### Distributed Options

```python
Distributed(actor="A100", rollout="A10G", num_rollout_workers=2)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `actor` | `"A100"` | GPU for actor (training) worker |
| `rollout` | `"A10G"` | GPU for rollout (generation) workers |
| `num_rollout_workers` | `2` | Number of parallel rollout workers |
| `weight_sync_method` | `"reload"` | How to sync weights (see below) |
| `sync_weights_every` | `1` | Sync frequency (every N steps) |

### Weight Sync Methods

```
                    Speed           How it works
                    ─────           ────────────
  reload (default)  ~2-3s           vLLM v1 sleep/wake_up/reload_weights
  volume            ~10-20s         Save to shared Modal volume, workers reload
  direct            ~2-3s           In-memory transfer via vLLM load_weights
  checkpoint        ~30-40s         Full checkpoint save + model recreation
```

---

## File Structure

```
mortal/
├── __init__.py              # Exports: MortalTrainer, SingleNode, Distributed, GPUConfig
├── app.py                   # Modal app, images (TRAINING_IMAGE, VLLM_IMAGE), volume
├── config.py                # OrchestratorConfig, SingleNode, Distributed, GPUConfig
├── orchestrator.py          # train() [Modal], train_local(), SingleNodeTrainer
├── trainer.py               # MortalTrainer — programmatic entry point
├── rewards/
│   ├── __init__.py          # Exports: RewardEnvironment, SandboxConfig, FunctionConfig
│   ├── base.py              # RewardEnvironment ABC, SandboxConfig, FunctionConfig
│   ├── dispatch.py          # compute_rewards() — routes to correct backend
│   ├── sandbox_executor.py  # Modal Sandbox execution (isolated containers)
│   ├── function_executor.py # Modal Function execution (pre-warmed TRAINING_IMAGE)
│   ├── utils.py             # Code extraction helpers
│   └── examples/
│       ├── code_execution.py    # CodeExecutionEnvironment
│       └── llm_judge.py         # LLMJudgeEnvironment
└── workers/
    ├── actor.py             # ActorWorker — RL training, weight management
    ├── rollout.py           # RolloutWorker — vLLM generation, weight sync
    └── reward.py            # Legacy reward helpers

examples/
└── gsm8k_grpo.py            # GSM8K math training — all 4 modes
```

---

## Testing

```bash
# Unit tests (reward environments only, no GPU needed)
python tests/test_trainer.py --unit-only

# Training tests (full loop with Modal GPU workers)
python tests/test_trainer.py --train-only

# All tests
python tests/test_trainer.py
```

---

## Monitoring

| Tool | What it shows |
|------|---------------|
| **[Modal Dashboard](https://modal.com/apps)** | Function calls, sandboxes, logs, GPU usage, costs |
| **Weights & Biases** | Training metrics: loss, reward, KL divergence, clip fraction |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of Memory** | Reduce `batch_size`, `max_completion_length`, or `num_generations`. Enable LoRA (`use_lora=True`). |
| **Rewards all zero** | Check `max_completion_length` -- if too small, completions get clipped before useful output. Qwen3 models use `<think>` blocks that consume tokens. Set to 2048+. |
| **Slow generation** | Increase `num_rollout_workers`. Enable vLLM in SingleNode. Reduce `max_tokens`. |
| **Weight sync fails** | Default `reload` requires vLLM v1 (>= 0.8.0). Fall back to `weight_sync_method="volume"`. |
| **Sandboxes not on dashboard** | Sandboxes spawned inside containers may not appear. Use `execute_in_function()` instead -- always visible as function calls. |
| **Function not hydrated** | Import `mortal.rewards.function_executor` before `app.run()` so Modal discovers the function. |

---

## References

- [GRPO Paper (DeepSeek)](https://arxiv.org/abs/2402.03300) -- Group Relative Policy Optimization
- [TRL Documentation](https://huggingface.co/docs/trl) -- Training framework
- [vLLM Documentation](https://docs.vllm.ai) -- Fast inference engine
- [Modal Documentation](https://modal.com/docs) -- Serverless GPU platform
- [veRL](https://github.com/volcengine/verl) -- Architecture inspiration
