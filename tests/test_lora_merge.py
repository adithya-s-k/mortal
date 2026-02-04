"""Test LoRA weight merging for vLLM compatibility."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def test_lora_merge():
    """Test that LoRA weights can be merged and produce correct key names for vLLM."""
    print("=== Testing LoRA Weight Merging ===\n")

    # Use a tiny model for fast testing
    model_name = "sshleifer/tiny-gpt2"

    print(f"1. Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get base model keys for comparison
    base_keys = set(base_model.state_dict().keys())
    print(f"   Base model has {len(base_keys)} parameters")
    print(f"   Sample keys: {list(base_keys)[:3]}")

    print("\n2. Adding LoRA adapters")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    # Get LoRA model keys
    lora_keys = set(peft_model.state_dict().keys())
    print(f"   LoRA model has {len(lora_keys)} parameters")
    print(f"   Sample keys: {list(lora_keys)[:5]}")

    # Check for base_model prefix
    has_base_model_prefix = any(k.startswith("base_model.") for k in lora_keys)
    print(f"   Has 'base_model.' prefix: {has_base_model_prefix}")

    print("\n3. Testing merge_adapter / unmerge_adapter")

    # Before merge - check a weight value
    test_key = list(lora_keys)[0]
    if "lora" not in test_key:
        # Find a base weight
        for k in lora_keys:
            if "lora" not in k and "weight" in k:
                test_key = k
                break

    # Merge adapters
    print("   Calling merge_adapter()...")
    peft_model.merge_adapter()

    # Get base model with merged weights
    print("   Getting base model state dict...")
    merged_base = peft_model.get_base_model()
    merged_state_dict = merged_base.state_dict()
    merged_keys = set(merged_state_dict.keys())

    print(f"   Merged model has {len(merged_keys)} parameters")
    print(f"   Sample merged keys: {list(merged_keys)[:3]}")

    # Check if keys match original base model format
    keys_match = merged_keys == base_keys
    print(f"   Keys match base model format: {keys_match}")

    if not keys_match:
        extra_keys = merged_keys - base_keys
        missing_keys = base_keys - merged_keys
        if extra_keys:
            print(f"   Extra keys: {list(extra_keys)[:5]}")
        if missing_keys:
            print(f"   Missing keys: {list(missing_keys)[:5]}")

    # Unmerge to restore LoRA for continued training
    print("\n   Calling unmerge_adapter()...")
    peft_model.unmerge_adapter()

    # Verify model still works after unmerge
    print("   Verifying model works after unmerge...")
    test_input = tokenizer("Hello", return_tensors="pt")
    with torch.no_grad():
        output = peft_model(**test_input)
    print(f"   Output shape: {output.logits.shape} ✓")

    print("\n4. Simulating weight save for vLLM")

    # This is what the actor does when saving weights
    def get_vllm_compatible_weights(model, use_lora=True):
        """Extract weights compatible with vLLM (no base_model prefix, no LoRA keys)."""
        from peft import PeftModel

        if use_lora and isinstance(model, PeftModel):
            print("   Using merge_and_unload for clean weights...")
            # merge_and_unload gives us a clean model without LoRA modules
            # But it's destructive, so we need to handle this carefully

            # Alternative: manually extract and merge weights
            # Get the base model weights with LoRA merged in
            model.merge_adapter()

            # Get state dict and filter/rename keys
            raw_state_dict = model.state_dict()
            clean_state_dict = {}

            for key, value in raw_state_dict.items():
                # Skip LoRA-specific parameters
                if "lora_" in key:
                    continue

                # Remove base_model.model. prefix
                clean_key = key.replace("base_model.model.", "")

                # Remove .base_layer from key names (LoRA wraps original layers)
                clean_key = clean_key.replace(".base_layer", "")

                clean_state_dict[clean_key] = value

            model.unmerge_adapter()
            return clean_state_dict
        else:
            return model.state_dict()

    vllm_weights = get_vllm_compatible_weights(peft_model, use_lora=True)
    vllm_keys = set(vllm_weights.keys())

    print(f"   vLLM-compatible weights: {len(vllm_keys)} parameters")
    print(f"   Sample keys: {list(vllm_keys)[:3]}")

    # Final check - no base_model prefix
    has_bad_prefix = any(k.startswith("base_model.") for k in vllm_keys)
    print(f"   Has 'base_model.' prefix (should be False): {has_bad_prefix}")

    # Check if vLLM keys match base model keys
    vllm_keys_match_base = vllm_keys == base_keys
    print(f"   vLLM keys match base model: {vllm_keys_match_base}")

    if not vllm_keys_match_base:
        extra = vllm_keys - base_keys
        missing = base_keys - vllm_keys
        if extra:
            print(f"   Extra keys in vLLM weights: {list(extra)[:3]}")
        if missing:
            print(f"   Missing from vLLM weights: {list(missing)[:3]}")

    print("\n=== Test Results ===")
    all_passed = True

    if has_bad_prefix:
        print("❌ FAILED: vLLM weights still have 'base_model.' prefix")
        all_passed = False
    else:
        print("✓ vLLM weights have correct key format (no 'base_model.' prefix)")

    if not vllm_keys_match_base:
        print("❌ FAILED: vLLM keys don't match base model format")
        all_passed = False
    else:
        print("✓ vLLM keys exactly match base model format")

    if all_passed:
        print("\n✅ All tests passed! LoRA merging works correctly.")
    else:
        print("\n❌ Some tests failed.")

    return all_passed


if __name__ == "__main__":
    test_lora_merge()
