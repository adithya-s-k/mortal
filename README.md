# MRL - Modal Reinforcement Learning

A serverless reinforcement learning framework for LLMs on [Modal](https://modal.com). Built on TRL and vLLM, MRL supports any RL algorithm available in TRL (GRPO, PPO, DPO, etc.) with pluggable reward functions, configurable GPU allocation, and a simple programmatic API. Currently ships with GRPO as the default training algorithm.

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
|  Reward Compute     |      |                              |
|  (local CPU or      |      |  Sandbox (if sandbox mode)   |
|   Modal Sandbox)    |----->|  +- code execution           |
|                     |      |                              |
+---------------------+      +------------------------------+
```

- **ActorWorker**: RL training step on GPU (A100 by default, configurable)
- **RolloutWorker**: vLLM inference on GPU (A10G by default, configurable)
- **Reward**: Modal Sandbox (code execution) or custom CPU functions

**Two orchestration modes:**
- `reward_funcs=None/"sandbox"` -- orchestrator + rewards run on Modal
- `reward_funcs=callable` -- orchestrator + rewards run locally, GPU work on Modal

## Quick Start

### Prerequisites

```bash
pip install modal
modal setup  # authenticate
modal volume create grpo-trl-storage  # one-time
```

### CLI Usage

```bash
# Default training (sandbox code-execution rewards)
modal run MRL/train.py::main --max-steps 5

# With GPU and worker options
modal run MRL/train.py::main \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-steps 10 \
  --batch-size 4 \
  --num-rollout-workers 2 \
  --actor-gpu A100 \
  --rollout-gpu A10G

# Minimal test run
modal run MRL/train.py::main \
  --max-steps 1 --batch-size 2 --max-samples 4 \
  --num-rollout-workers 1 --num-generations 2

# Background run
modal run --detach MRL/train.py::main --max-steps 100
```

### Programmatic API (MRLTrainer)

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
    actor_gpu="A100",
    rollout_gpu="A10G",
)
trainer.train()
```

#### Multiple Reward Functions

When multiple reward functions are provided, their scores are averaged per completion.

```python
def format_reward(completions, **kwargs):
    return [1.0 if "```" in c else 0.0 for c in completions]

def length_reward(completions, **kwargs):
    return [min(len(c) / 500, 1.0) for c in completions]

trainer = MRLTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[format_reward, length_reward],
    train_dataset=dataset,
)
trainer.train()
```

## Reward Functions

MRL supports three reward modes:

| Mode | `reward_funcs=` | Where it runs | Use case |
|------|-----------------|---------------|----------|
| Sandbox | `None` or `"sandbox"` | Modal Sandbox | Code execution with test cases |
| Custom | `callable` | Local CPU | Any custom reward logic |
| Multi | `[callable, ...]` | Local CPU | Multiple rewards, averaged |

Custom reward functions receive `completions` as a list of strings plus any extra dataset columns as keyword arguments:

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
| `reward_funcs` | `None` | Reward function(s) or `"sandbox"` |
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

### CLI Arguments

All MRLTrainer parameters are available as CLI args via `modal run MRL/train.py::main`. Additional CLI-only options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--simple-mode` | `False` | Use TRL's built-in GRPOTrainer loop |

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
|     Example: 4 coding problems                                      |
|                          |                                          |
|                          v                                          |
|  2. GENERATE COMPLETIONS (Parallel)                                 |
|     Each prompt is sent to rollout workers                          |
|     With num_generations=2, we get 8 completions total              |
|     +-------------+ +-------------+                                 |
|     |RolloutWorker| |RolloutWorker|                                 |
|     |  4 prompts  | |  4 prompts  |                                 |
|     |  -> 4 codes | |  -> 4 codes |                                 |
|     +------+------+ +------+------+                                 |
|            +-------+-------+                                        |
|                    v                                                |
|  3. COMPUTE REWARDS                                                 |
|     Sandbox mode: execute code in Modal Sandboxes (parallel)        |
|     Custom mode: run reward function(s) on local CPU                |
|     +--------+ +--------+ +--------+ +--------+                    |
|     |Sandbox1| |Sandbox2| |  ...   | |Sandbox8|                    |
|     | r=1    | | r=0    | |        | | r=1    |                    |
|     +---+----+ +---+----+ +---+----+ +---+----+                    |
|         +----------+-----------+----------+                         |
|                    v                                                |
|  4. TRAIN STEP                                                      |
|     ActorWorker receives (prompts, completions, rewards)            |
|     Computes RL loss and updates model weights                      |
|                    |                                                |
|                    v                                                |
|  5. SYNC WEIGHTS (Every N steps)                                    |
|     Actor saves weights to shared volume                            |
|     RolloutWorkers reload via sleep/wake_up/reload_weights (~2s)    |
|     Now generating with updated policy!                             |
|                    |                                                |
|                    v                                                |
|  6. REPEAT until max_steps reached                                  |
|                                                                     |
+---------------------------------------------------------------------+
```

### Default Algorithm: GRPO

The default training algorithm is GRPO (Group Relative Policy Optimization), which generates multiple completions per prompt and computes relative advantages within each group:

1. Generate `num_generations` completions per prompt
2. Score each with the reward function
3. Normalize rewards within each group (zero-mean, unit-variance)
4. Update policy to increase probability of higher-reward completions

Supported loss variants: `dapo`, `grpo`, `dr_grpo`, `bnpo`, `cispo`, `sapo`

Other TRL-supported algorithms (PPO, DPO, etc.) can be integrated by swapping the ActorWorker's training logic.

## File Structure

```
MRL/
+-- __init__.py          # Package exports (MRLTrainer, configs)
+-- app.py               # Modal app, images, volume definitions
+-- config.py            # OrchestratorConfig, ModelConfig, TrainingConfig, GenerationConfig
+-- rewards.py           # Reward dispatch layer (sandbox vs custom)
+-- orchestrator.py      # train() [Modal], train_local() [local], training loop
+-- trainer.py           # MRLTrainer facade (programmatic API)
+-- train.py             # CLI entry point (modal run)
+-- workers/
    +-- actor.py         # ActorWorker - RL training, weight management, checkpointing
    +-- rollout.py       # RolloutWorker - vLLM generation, weight sync
    +-- reward.py        # Sandbox-based code execution rewards
```

## Testing Individual Components

```bash
# Test rollout worker generation
modal run MRL/train.py::test_rollout_fn

# Test reward computation
modal run MRL/train.py::test_reward_fn

# List saved checkpoints
modal run MRL/train.py::list_checkpoints_fn

# Clear corrupted model cache
modal run MRL/train.py::clear_cache
modal run MRL/train.py::clear_cache --model Qwen/Qwen2-0.5B-Instruct
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
- Use `modal run MRL/train.py::clear_cache` to reset corrupted model cache

## References

- [GRPO Paper (DeepSeek)](https://arxiv.org/abs/2402.03300)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [vLLM Documentation](https://docs.vllm.ai)
- [Modal Documentation](https://modal.com/docs)
- [veRL](https://github.com/volcengine/verl) - Architecture inspiration
