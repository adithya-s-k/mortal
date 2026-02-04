# MRL - Modal Reinforcement Learning

A veRL-style serverless architecture for GRPO (Group Relative Policy Optimization) training on [Modal](https://modal.com), using TRL's GRPOTrainer and standalone vLLM workers.

## What is GRPO?

**GRPO (Group Relative Policy Optimization)** is a reinforcement learning algorithm introduced by DeepSeek that improves language models through reward-based learning. Unlike traditional RL methods that require a separate critic network, GRPO uses a simpler approach:

1. **Generate multiple completions** for each prompt (e.g., 4 different responses)
2. **Score each completion** using a reward function (e.g., code execution tests)
3. **Compute relative advantages** - compare completions within the same group
4. **Update the policy** to increase probability of higher-reward completions

The loss function is:
```
L = -E[advantage * log_prob(completion)]
```

Where the advantage is computed relative to other completions in the same group, eliminating the need for a baseline/critic network.

## Why This Architecture?

Traditional RL training runs everything on a single GPU, which creates bottlenecks:
- **Generation is slow** on training GPUs (not optimized for inference)
- **Reward computation blocks training** while waiting for evaluations
- **No horizontal scaling** - can't add more inference capacity

This framework solves these problems with a **distributed, serverless architecture**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MODAL CLOUD                                   │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    Orchestrator (CPU)                              │ │
│  │  • Loads dataset and manages training loop                        │ │
│  │  • Coordinates all workers                                        │ │
│  │  • Logs metrics to Weights & Biases                               │ │
│  └───────────────────────┬───────────────────────────────────────────┘ │
│                          │                                              │
│          ┌───────────────┼───────────────┬───────────────┐             │
│          ▼               ▼               ▼               ▼             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────┐      │
│  │ ActorWorker  │ │RolloutWorker │ │RolloutWorker │ │ Reward   │      │
│  │   (A100)     │ │   (A10G)     │ │   (A10G)     │ │ Workers  │      │
│  │ TRL Trainer  │ │    vLLM      │ │    vLLM      │ │(Sandbox) │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────┘      │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Shared Modal Volume                            │   │
│  │  /storage/checkpoints    /storage/hf_cache                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- **Parallel generation**: Multiple vLLM workers generate completions simultaneously
- **Fast inference**: vLLM is optimized for inference (continuous batching, PagedAttention)
- **Secure execution**: Code runs in isolated Modal Sandboxes
- **Serverless scaling**: Pay only for compute you use, scale to zero when idle
- **Efficient weight sync**: Uses vLLM v1's `sleep/wake_up/reload_weights` for ~150x faster updates

## Quick Start

### Prerequisites

1. **Install Modal CLI:**
   ```bash
   pip install modal
   modal setup  # Authenticate with Modal
   ```

2. **Create Modal secrets** (for HuggingFace and Weights & Biases):
   ```bash
   modal secret create hf-wandb-secret \
     HF_TOKEN=<your-huggingface-token> \
     WANDB_API_KEY=<your-wandb-key>
   ```

   > **Note:** Update the secret name in `app.py` if you use a different name.

3. **Create the storage volume:**
   ```bash
   modal volume create grpo-trl-storage
   ```

### Run Training

```bash
# Basic training run
modal run MRL/train.py

# With custom parameters
modal run MRL/train.py \
  --max-samples 100 \
  --max-steps 10 \
  --batch-size 4 \
  --num-rollout-workers 2 \
  --num-generations 2 \
  --max-tokens 1024 \
  --max-model-len 4096 \
  --sync-weights-every 5

# Run in background (detached)
modal run --detach MRL/train.py
```

### Monitor Training

- **Modal Dashboard**: https://modal.com/apps - View logs, costs, GPU usage
- **Weights & Biases**: Training metrics logged to project `modal-grpo-trl`

## How It Works: Step by Step

### The Training Loop

Each training step follows this flow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. GET BATCH                                                        │
│     Orchestrator fetches batch of prompts from dataset               │
│     Example: 4 coding problems                                       │
│                          │                                           │
│                          ▼                                           │
│  2. GENERATE COMPLETIONS (Parallel)                                  │
│     Each prompt is sent to rollout workers                           │
│     With num_generations=2, we get 8 completions total               │
│     ┌─────────────┐ ┌─────────────┐                                 │
│     │RolloutWorker│ │RolloutWorker│                                 │
│     │  4 prompts  │ │  4 prompts  │                                 │
│     │  → 4 codes  │ │  → 4 codes  │                                 │
│     └──────┬──────┘ └──────┬──────┘                                 │
│            └───────┬───────┘                                         │
│                    ▼                                                 │
│  3. COMPUTE REWARDS (Parallel via Sandboxes)                         │
│     Each completion is executed against test cases                   │
│     Reward = 1 if all tests pass, 0 otherwise                        │
│     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                 │
│     │Sandbox 1│ │Sandbox 2│ │   ...   │ │Sandbox 8│                 │
│     │ Code 1  │ │ Code 2  │ │         │ │ Code 8  │                 │
│     │ r=1     │ │ r=0     │ │         │ │ r=1     │                 │
│     └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘                 │
│          └───────────┴───────────┴───────────┘                       │
│                          │                                           │
│                          ▼                                           │
│  4. TRAIN STEP                                                       │
│     ActorWorker receives (prompts, completions, rewards)             │
│     Computes GRPO loss and updates model weights                     │
│     Loss = -mean(reward * log_prob(completion))                      │
│                          │                                           │
│                          ▼                                           │
│  5. SYNC WEIGHTS (Every N steps)                                     │
│     Actor saves weights to shared volume                             │
│     RolloutWorkers use vLLM reload_weights (~2s vs ~40s full reload) │
│     Now generating with updated policy!                              │
│                          │                                           │
│                          ▼                                           │
│  6. REPEAT until max_steps reached                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Example: Training on Code Generation

The default dataset is `OpenCoder-LLM/opc-sft-stage2` with coding problems:

**Input prompt:**
```
Write a Python function to check if a number is prime.
```

**Model generates multiple completions:**
```python
# Completion 1 (reward=1, passes tests)
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Completion 2 (reward=0, fails tests)
def is_prime(n):
    return n > 1  # Too simple, fails edge cases
```

**Test cases executed in sandboxes:**
```python
assert is_prime(2) == True
assert is_prime(4) == False
assert is_prime(17) == True
assert is_prime(1) == False
```

**GRPO Update:**
- Completion 1 gets positive advantage (reward above group mean)
- Completion 2 gets negative advantage (reward below group mean)
- Model learns to generate code more like Completion 1

## Project Structure

```
MRL/
├── __init__.py          # Package exports
├── app.py               # Modal app, images, and volume definitions
├── config.py            # Configuration dataclasses
├── orchestrator.py      # Main training loop coordinator
├── train.py             # CLI entry point
├── workers/
│   ├── __init__.py      # Worker exports
│   ├── actor.py         # ActorWorker - TRL GRPOTrainer
│   ├── rollout.py       # RolloutWorker - vLLM inference
│   └── reward.py        # Reward computation via Sandboxes
└── tests/
    ├── __init__.py      # Test package
    ├── test_vllm.py     # vLLM API and weight update tests
    └── test_weight_sync.py  # Weight sync workflow tests
```

## Components in Detail

### 1. ActorWorker (`workers/actor.py`)

The training worker that updates model weights using TRL's GRPOTrainer.

| Property | Value |
|----------|-------|
| GPU | A100 (configurable) |
| Framework | TRL GRPOTrainer |
| Role | Policy optimization |

**What it does:**
1. Loads the base model (e.g., Qwen2-0.5B-Instruct)
2. Receives (prompts, completions, rewards) from orchestrator
3. Computes GRPO loss: `loss = -mean(rewards * log_probs)`
4. Updates model weights via backpropagation
5. Saves checkpoints to shared volume

**Key Methods:**
```python
# Initialize the trainer
actor.initialize(config)

# Single training step with pre-computed data
actor.train_step(prompts, completions, rewards, logprobs)

# Save checkpoint for weight sync
actor.save_checkpoint(step)
```

### 2. RolloutWorker (`workers/rollout.py`)

Fast inference worker using standalone vLLM for generation.

| Property | Value |
|----------|-------|
| GPU | A10G (24GB) |
| Framework | vLLM |
| Idle Timeout | 300s (stays warm) |

**What it does:**
1. Loads model into vLLM engine (optimized for inference)
2. Generates completions for batches of prompts
3. Returns completions with log probabilities
4. Reloads from checkpoints when weights are synced

**Why vLLM?**
- **Continuous batching**: Processes multiple requests efficiently
- **PagedAttention**: Better memory management
- **3-5x faster** than naive HuggingFace generation

**Key Methods:**
```python
# Generate completions
result = rollout.generate(
    prompts=["Write a function..."],
    model_path="Qwen/Qwen2-0.5B-Instruct",
    max_tokens=1024,
    temperature=0.7
)
# Returns: {"completions": [...], "logprobs": [...]}

# Efficient weight update (uses vLLM v1 sleep/wake_up/reload_weights)
rollout.update_weights_from_volume("/storage/model_cache/Qwen_Qwen2-0.5B-Instruct")

# Or reload from checkpoint (slower, recreates model)
rollout.reload_from_checkpoint("/storage/checkpoints/step-5")
```

### 3. Reward Workers (`workers/reward.py`)

Secure code execution in Modal Sandboxes.

| Property | Value |
|----------|-------|
| Execution | Modal Sandbox (isolated) |
| Timeout | 30s per execution |
| Security | No network, limited filesystem |

**What it does:**
1. Extracts code from model completion (handles ```python blocks)
2. Combines code with test cases
3. Executes in isolated sandbox
4. Returns reward: 1 if all tests pass, 0 otherwise

**Why Sandboxes?**
- **Security**: Untrusted code runs in isolation
- **Parallelism**: Each execution is independent
- **No side effects**: Clean environment for each test

**Key Functions:**
```python
# Binary reward (pass/fail)
reward = compute_reward(completion, testcases)  # Returns 0 or 1

# Partial credit (proportion of tests passed)
reward = compute_reward_with_partial_credit(completion, testcases)  # Returns 0.0-1.0

# TRL-compatible batch function
rewards = reward_helper_function(completions, testcases)  # Returns list
```

### 4. Orchestrator (`orchestrator.py`)

Coordinates all workers in the training loop.

**What it does:**
1. Loads dataset and initializes workers
2. For each training step:
   - Fetches batch of prompts
   - Distributes generation across rollout workers
   - Collects completions and computes rewards
   - Sends data to actor for training
   - Periodically syncs weights to rollout workers
3. Logs metrics to Weights & Biases
4. Saves final checkpoint

## Configuration

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen2-0.5B-Instruct` | HuggingFace model name |
| `--epochs` | 5 | Number of training epochs |
| `--max-steps` | 5 | Maximum training steps (-1 for unlimited) |
| `--batch-size` | 8 | Prompts per training step |
| `--num-rollout-workers` | 2 | Number of vLLM workers |
| `--num-generations` | 4 | Completions per prompt |
| `--max-samples` | 128 | Dataset samples (0 for full) |
| `--max-tokens` | 8000 | Max tokens per completion |
| `--max-model-len` | 16384 | Model context length |
| `--learning-rate` | 5e-6 | Learning rate |
| `--save-steps` | 100 | Checkpoint frequency |
| `--sync-weights-every` | 1 | Weight sync frequency |
| `--weight-sync-method` | `reload` | Weight sync method (see below) |
| `--simple-mode` | False | Use TRL's built-in loop |

### Weight Sync Methods

Weight synchronization keeps rollout workers updated with the latest trained weights. We support multiple methods optimized for different scenarios:

| Method | Speed | Description |
|--------|-------|-------------|
| `reload` (default) | **Fast** (~2-3s) | Uses vLLM v1's `sleep/wake_up/reload_weights` pattern. No model recreation. |
| `volume` | Medium (~10-20s) | Saves to shared volume, workers reload from volume (recreates model). |
| `checkpoint` | Slow (~30-40s) | Full checkpoint save + model recreation. Most reliable. |

**The `reload` method** is recommended and uses vLLM's efficient weight update API:
1. Actor saves weights to the model cache path on the shared volume
2. Rollout workers call `sleep(level=2)` to free GPU memory
3. Workers call `reload_weights()` to load updated weights in-place
4. Workers call `wake_up()` to reallocate KV cache

This achieves **~150x speedup** over full model recreation (~0.3s reload vs ~45s full init).

### Recommended Configurations

**Quick test run:**
```bash
modal run MRL/train.py \
  --max-samples 50 \
  --max-steps 5 \
  --batch-size 4 \
  --num-generations 2 \
  --sync-weights-every 100  # Disable sync for speed
```

**Full training run:**
```bash
modal run MRL/train.py \
  --max-samples 0 \           # Full dataset
  --max-steps 1000 \
  --batch-size 8 \
  --num-rollout-workers 4 \   # More parallel generation
  --num-generations 4 \
  --sync-weights-every 10     # Sync every 10 steps
```

## Volume Structure

All persistent data is stored on a shared Modal volume:

```
/storage/
├── checkpoints/
│   ├── step-5/           # Checkpoint at step 5
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer files...
│   ├── step-10/
│   └── ...
└── hf_cache/             # HuggingFace model cache
    └── hub/
        └── models--Qwen--Qwen2-0.5B-Instruct/
```

**Inspect volume contents:**
```bash
modal volume ls grpo-trl-storage
modal volume ls grpo-trl-storage checkpoints/
```

## Testing Individual Components

```bash
# Test rollout worker generation
modal run MRL/train.py::test_rollout_fn

# Test reward computation
modal run MRL/train.py::test_reward_fn

# List saved checkpoints
modal run MRL/train.py::list_checkpoints_fn
```

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** CUDA OOM errors during training or generation

**Solutions:**
- Reduce `--batch-size`
- Reduce `--max-model-len`
- Reduce `--num-generations`
- Use a smaller model

### Slow Generation

**Symptoms:** Long wait times during completion generation

**Solutions:**
- Increase `--num-rollout-workers` (more parallel generation)
- Reduce `--max-tokens` (shorter completions)
- Check Modal dashboard for GPU utilization

### Weight Sync Issues

**Symptoms:** Rollout workers using stale weights

**How weight sync works (reload method - default):**
1. Actor saves weights to `/storage/model_cache/<model_name>/model.safetensors`
2. Actor calls `volume.commit()` to sync to shared storage
3. Rollout workers use vLLM v1's `sleep/wake_up/reload_weights` pattern
4. Weights are loaded in-place without model recreation

**Debug:**
```bash
# Check if model cache exists
modal volume ls grpo-trl-storage model_cache/

# Check weight sync manifest
modal volume get grpo-trl-storage model_cache/Qwen_Qwen2-0.5B-Instruct/sync_manifest.json
```

**If reload fails, try:**
- Use `--weight-sync-method volume` or `--weight-sync-method checkpoint` as fallback
- Check vLLM version supports v1 API (requires vLLM >= 0.8.0)

### Low Rewards

**Symptoms:** Mean reward stays at 0 or very low

**Possible causes:**
- Model too small for the task
- Prompts too difficult
- Test cases too strict
- Temperature too low (not enough exploration)

**Solutions:**
- Try a larger model
- Use `--max-samples` to filter easier examples
- Increase `--temperature` (default 0.7)
- Use partial credit rewards

## Cost Estimation

Modal pricing (approximate, check modal.com for current rates):

| Resource | Cost | Usage |
|----------|------|-------|
| A100 GPU | ~$2.50/hr | Actor training |
| A10G GPU | ~$1.10/hr | Rollout workers (x2) |
| CPU | ~$0.10/hr | Orchestrator |
| Sandbox | ~$0.001/exec | Reward computation |

**Example 10-step training run:**
- Duration: ~8-10 minutes
- Estimated cost: ~$0.50-1.00

## Advanced: Customization

### Using a Different Dataset

Modify `orchestrator.py` or create a custom config:

```python
config = {
    "dataset_name": "your-org/your-dataset",
    "dataset_config": "default",
    "dataset_split": "train",
}
```

Dataset must have columns that map to:
- `prompt` (or rename via `dataset.rename_column()`)
- `testcases` (list of assert statements)

### Custom Reward Function

Modify `workers/reward.py`:

```python
def custom_reward(completion: str, metadata: dict) -> float:
    # Your custom logic here
    # Return float between 0 and 1
    pass
```

### Different Model

```bash
modal run MRL/train.py --model "meta-llama/Llama-2-7b-chat-hf"
```

Note: Larger models require more GPU memory. Adjust `--max-model-len` accordingly.

## References

- [GRPO Paper (DeepSeek)](https://arxiv.org/abs/2402.03300) - Original algorithm
- [TRL Documentation](https://huggingface.co/docs/trl) - Training library
- [vLLM Documentation](https://docs.vllm.ai) - Inference engine
- [Modal Documentation](https://modal.com/docs) - Serverless platform
- [veRL](https://github.com/volcengine/verl) - Inspiration for architecture

## License

MIT License
