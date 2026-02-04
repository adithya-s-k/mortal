"""Test script to debug vLLM weight updates on Modal."""

import modal

app = modal.App("test-vllm-weights")

# Test with v1 engine (default) - has collective_rpc for direct weight updates
vllm_image_v1 = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("pip install uv")
    .run_commands("uv pip install 'vllm>=0.8.0' 'transformers==4.57.6' hf_transfer accelerate safetensors --system")
    .env({
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        # VLLM_USE_V1 defaults to 1 (v1 engine)
    })
)

# Test with v0 engine (legacy) - uses direct model access
vllm_image_v0 = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm transformers hf_transfer accelerate --system")
    .env({
        "VLLM_USE_V1": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
)

volume = modal.Volume.from_name("grpo-trl-storage", create_if_missing=True)


@app.function(
    image=vllm_image_v1,
    gpu="A10G",
    volumes={"/storage": volume},
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
    timeout=600,
)
def test_vllm_v1():
    """Test vLLM v1 engine and weight update APIs."""
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    from vllm import LLM, SamplingParams
    import vllm

    print(f"vLLM version: {vllm.__version__}")

    print("\n=== Testing vLLM v1 Engine ===")
    print("Creating LLM...")
    llm = LLM(
        model="Qwen/Qwen2-0.5B-Instruct",
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        enforce_eager=True,
    )
    print("LLM created successfully!")

    # Check v1 API availability
    has_collective_rpc = hasattr(llm, "collective_rpc")
    has_sleep = hasattr(llm, "sleep")
    has_wake_up = hasattr(llm, "wake_up")

    print(f"\nv1 API Check:")
    print(f"  has collective_rpc: {has_collective_rpc}")
    print(f"  has sleep: {has_sleep}")
    print(f"  has wake_up: {has_wake_up}")

    is_v1 = has_collective_rpc and has_sleep and has_wake_up
    print(f"  Is v1 engine: {is_v1}")

    # Test basic generation first
    print("\n=== Testing Generation ===")
    prompts = ["Write a Python function to add two numbers:"]
    sampling_params = SamplingParams(max_tokens=100, temperature=0.7)

    outputs = llm.generate(prompts, sampling_params)
    print(f"Generated: {outputs[0].outputs[0].text[:100]}...")

    # Test weight update API if v1
    if is_v1:
        print("\n=== Testing v1 Weight Update API ===")
        try:
            # Inspect what reload_weights expects
            import inspect
            try:
                # Try to get the worker's reload_weights signature
                from vllm.worker.gpu_worker import Worker
                sig = inspect.signature(Worker.reload_weights)
                print(f"Worker.reload_weights signature: {sig}")
            except Exception as e:
                print(f"Could not inspect Worker.reload_weights: {e}")

            try:
                from vllm.v1.worker.gpu_worker import Worker as V1Worker
                sig = inspect.signature(V1Worker.reload_weights)
                print(f"V1Worker.reload_weights signature: {sig}")
            except Exception as e:
                print(f"Could not inspect V1Worker.reload_weights: {e}")

            # Get model weights
            from transformers import AutoModelForCausalLM
            import torch

            print("Loading HuggingFace model to get weights...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-0.5B-Instruct",
                torch_dtype=torch.float16,
                device_map="cpu",  # Load to CPU first
            )
            state_dict = hf_model.state_dict()
            print(f"Got {len(state_dict)} parameters from HF model")

            # Save weights to temp file (generators can't be serialized over RPC)
            from safetensors.torch import save_file
            import os

            temp_weights_path = "/storage/test_weights"
            os.makedirs(temp_weights_path, exist_ok=True)
            weights_file = f"{temp_weights_path}/model.safetensors"

            # Clone tensors to avoid shared memory issues with safetensors
            state_dict_clone = {k: v.clone().contiguous() for k, v in state_dict.items()}

            print(f"Saving weights to {weights_file}...")
            save_file(state_dict_clone, weights_file)
            del state_dict_clone
            print("  OK")

            # Test sleep/wake_up/reload_weights pattern
            print("\nTesting sleep(level=2)...")
            llm.sleep(level=2)
            print("  OK")

            print("Testing wake_up(tags=['weights'])...")
            llm.wake_up(tags=["weights"])
            print("  OK")

            # Try different reload_weights approaches
            print("\n--- Trying reload_weights approaches ---")

            reload_success = False

            # Approach 1: Try with weights_path (newer vLLM versions)
            try:
                print(f"Approach 1: collective_rpc('reload_weights', weights_path={weights_file})...")
                llm.collective_rpc("reload_weights", kwargs={"weights_path": weights_file})
                print("  OK - weights_path works!")
                reload_success = True
            except Exception as e:
                if "unexpected keyword argument" in str(e):
                    print(f"  weights_path not supported in this version")
                else:
                    print(f"  Failed with: {e}")

            # Approach 2: Try without arguments (reloads from original model)
            if not reload_success:
                try:
                    print("Approach 2: collective_rpc('reload_weights') - no args (reloads original model)...")
                    llm.collective_rpc("reload_weights")
                    print("  OK - no-args reload works!")
                    reload_success = True
                except Exception as e:
                    print(f"  Failed with: {e}")

            # Approach 3: Try with weights_iterator (if supported)
            if not reload_success:
                try:
                    print("Approach 3: collective_rpc('reload_weights', weights_iterator=...)...")

                    # Create iterator from state_dict
                    def make_weights_iter():
                        for name, param in state_dict.items():
                            yield (name, param.to("cuda").contiguous())

                    llm.collective_rpc("reload_weights", kwargs={"weights_iterator": make_weights_iter()})
                    print("  OK - weights_iterator works!")
                    reload_success = True
                except Exception as e:
                    print(f"  Failed with: {e}")

            # Approach 4: Direct model access (bypass collective_rpc)
            if not reload_success:
                try:
                    print("Approach 4: Direct model runner access...")

                    # Try to access the model runner directly
                    # v1 engine structure: llm.llm_engine.engine_core -> workers -> model_runner
                    engine = llm.llm_engine

                    # Check different paths
                    model_runner = None
                    paths_tried = []

                    # Try v1 engine paths
                    try:
                        # For single process
                        core = engine.engine_core
                        if hasattr(core, 'model_executor'):
                            executor = core.model_executor
                            if hasattr(executor, 'driver_worker'):
                                model_runner = executor.driver_worker.model_runner
                                paths_tried.append("engine_core.model_executor.driver_worker.model_runner")
                    except Exception as e:
                        paths_tried.append(f"v1 path failed: {e}")

                    if model_runner is None:
                        try:
                            # Alternative: access via workers
                            if hasattr(engine, 'model_executor'):
                                executor = engine.model_executor
                                if hasattr(executor, 'driver_worker'):
                                    model_runner = executor.driver_worker.model_runner
                                    paths_tried.append("model_executor.driver_worker.model_runner")
                        except Exception as e:
                            paths_tried.append(f"alt path failed: {e}")

                    print(f"  Paths tried: {paths_tried}")

                    if model_runner is not None:
                        print(f"  Found model_runner: {type(model_runner)}")
                        sig = inspect.signature(model_runner.reload_weights)
                        print(f"  model_runner.reload_weights signature: {sig}")

                        # Check if it accepts weights_iterator
                        params = sig.parameters
                        if 'weights_iterator' in params:
                            print("  ✓ weights_iterator supported!")

                            # Create iterator
                            weights_list = [(name, param.to("cuda").contiguous())
                                          for name, param in state_dict.items()]

                            model_runner.reload_weights(weights_iterator=iter(weights_list))
                            print("  OK - Direct model_runner.reload_weights works!")
                            reload_success = True
                        else:
                            print(f"  ✗ weights_iterator not in params: {list(params.keys())}")
                    else:
                        print("  Could not find model_runner")

                except Exception as e:
                    print(f"  Failed with: {e}")
                    import traceback
                    traceback.print_exc()

            if not reload_success:
                print("\n⚠️ No reload_weights approach worked!")

            print("\nTesting wake_up(tags=['kv_cache'])...")
            llm.wake_up(tags=["kv_cache"])
            print("  OK")

            # Test generation after weight update
            print("\nTesting generation after weight update...")
            outputs = llm.generate(prompts, sampling_params)
            print(f"Generated: {outputs[0].outputs[0].text[:100]}...")

            print("\n✅ v1 weight update API works!")

            # Cleanup
            del hf_model
            del state_dict
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ v1 weight update failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️ v1 API not available, skipping weight update test")

    return {
        "status": "success",
        "vllm_version": vllm.__version__,
        "is_v1": is_v1,
        "has_collective_rpc": has_collective_rpc,
    }


@app.function(
    image=vllm_image_v0,
    gpu="A10G",
    volumes={"/storage": volume},
    secrets=[modal.Secret.from_name("adithya-hf-wandb")],
    timeout=600,
)
def test_vllm_v0():
    """Test vLLM v0 engine (legacy) and weight update APIs."""
    import os
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    from vllm import LLM, SamplingParams
    import vllm

    print(f"vLLM version: {vllm.__version__}")

    print("\n=== Testing vLLM v0 Engine (Legacy) ===")
    print("Creating LLM with VLLM_USE_V1=0...")
    llm = LLM(
        model="Qwen/Qwen2-0.5B-Instruct",
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        enforce_eager=True,
    )
    print("LLM created successfully!")

    # Check v0 API paths
    print("\n=== Checking v0 API Paths ===")
    paths_to_try = [
        ("llm.llm_engine.model_executor.driver_worker.model_runner.model",
         lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model),
        ("llm.llm_engine.model_executor.model",
         lambda: llm.llm_engine.model_executor.model),
        ("llm.llm_engine.driver_worker.model_runner.model",
         lambda: llm.llm_engine.driver_worker.model_runner.model),
    ]

    llm_model = None
    working_path = None
    for path_name, path_fn in paths_to_try:
        try:
            model = path_fn()
            if model is not None:
                has_load = hasattr(model, 'load_weights')
                print(f"  {path_name}: ✅ (has load_weights: {has_load})")
                if has_load and llm_model is None:
                    llm_model = model
                    working_path = path_name
            else:
                print(f"  {path_name}: ❌ (None)")
        except (AttributeError, TypeError) as e:
            print(f"  {path_name}: ❌ ({e})")

    # Test basic generation
    print("\n=== Testing Generation ===")
    prompts = ["Write a Python function to add two numbers:"]
    sampling_params = SamplingParams(max_tokens=100, temperature=0.7)

    outputs = llm.generate(prompts, sampling_params)
    print(f"Generated: {outputs[0].outputs[0].text[:100]}...")

    # Test weight update if we found a working path
    if llm_model is not None:
        print(f"\n=== Testing v0 Weight Update via {working_path} ===")
        try:
            from transformers import AutoModelForCausalLM
            import torch

            print("Loading HuggingFace model to get weights...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-0.5B-Instruct",
                torch_dtype=torch.float16,
                device_map="cpu",
            )

            # Create weights list
            weights_to_load = []
            for name, param in hf_model.state_dict().items():
                weights_to_load.append((name, param.to("cuda").contiguous()))
            print(f"Prepared {len(weights_to_load)} parameters")

            print("Calling load_weights...")
            llm_model.load_weights(weights_to_load)
            print("  OK")

            # Reset cache
            try:
                llm.reset_prefix_cache()
                print("Reset prefix cache: OK")
            except Exception as e:
                print(f"Reset prefix cache: {e}")

            # Test generation after update
            print("\nTesting generation after weight update...")
            outputs = llm.generate(prompts, sampling_params)
            print(f"Generated: {outputs[0].outputs[0].text[:100]}...")

            print("\n✅ v0 weight update API works!")

            del hf_model
            del weights_to_load
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ v0 weight update failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️ No working v0 API path found")

    return {
        "status": "success",
        "vllm_version": vllm.__version__,
        "working_path": working_path,
    }


@app.local_entrypoint()
def main(engine: str = "v1"):
    """Run the test.

    Args:
        engine: Which engine to test - "v1", "v0", or "both"
    """
    print(f"Testing vLLM on Modal (engine={engine})...")

    if engine in ("v1", "both"):
        print("\n" + "="*60)
        print("Testing v1 engine...")
        print("="*60)
        result = test_vllm_v1.remote()
        print(f"\nv1 Result: {result}")

    if engine in ("v0", "both"):
        print("\n" + "="*60)
        print("Testing v0 engine...")
        print("="*60)
        result = test_vllm_v0.remote()
        print(f"\nv0 Result: {result}")
