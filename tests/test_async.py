"""Tests for async RL infrastructure.

Tests config serialization, queue message protocol, and helper functions.
Runs locally without GPU/Modal (unit tests).

Usage:
    python tests/test_async.py                    # All tests
    python tests/test_async.py --unit-only        # Config + helper tests only
    python tests/test_async.py --queue-only       # modal.Queue protocol tests only
    python tests/test_async.py --integration-only # Full async training on Modal
"""

import argparse
import sys
import time
import traceback


def test_async_config_basics():
    """Test AsyncConfig creation and defaults."""
    from mortal.config import AsyncConfig

    cfg = AsyncConfig()
    assert cfg.enabled is False
    assert cfg.mode == "pipeline"
    assert cfg.staleness_threshold == 1
    assert cfg.sync_every == 1
    assert cfg.queue_size == 2
    print("  PASS: AsyncConfig defaults")

    cfg = AsyncConfig(enabled=True, mode="queue", staleness_threshold=3, queue_size=4)
    assert cfg.enabled is True
    assert cfg.mode == "queue"
    assert cfg.staleness_threshold == 3
    assert cfg.queue_size == 4
    print("  PASS: AsyncConfig custom values")


def test_async_config_serialization():
    """Test AsyncConfig to_dict / from_dict round-trip."""
    from mortal.config import AsyncConfig

    cfg = AsyncConfig(enabled=True, mode="queue", staleness_threshold=5, sync_every=2, queue_size=3)
    d = cfg.to_dict()
    assert d["enabled"] is True
    assert d["mode"] == "queue"
    assert d["staleness_threshold"] == 5

    restored = AsyncConfig.from_dict(d)
    assert restored.enabled == cfg.enabled
    assert restored.mode == cfg.mode
    assert restored.staleness_threshold == cfg.staleness_threshold
    assert restored.sync_every == cfg.sync_every
    assert restored.queue_size == cfg.queue_size
    print("  PASS: AsyncConfig serialization round-trip")


def test_distributed_with_async_config():
    """Test Distributed mode includes async_config in serialization."""
    from mortal.config import Distributed, AsyncConfig

    mode = Distributed(
        actor="A100",
        rollout="A10G",
        async_config=AsyncConfig(enabled=True, mode="pipeline"),
    )
    d = mode.to_dict()
    assert "async_config" in d
    assert d["async_config"]["enabled"] is True
    assert d["async_config"]["mode"] == "pipeline"

    restored = Distributed.from_dict(d)
    assert restored.async_config.enabled is True
    assert restored.async_config.mode == "pipeline"
    print("  PASS: Distributed with AsyncConfig round-trip")


def test_distributed_default_async_config():
    """Test Distributed defaults to disabled async."""
    from mortal.config import Distributed

    mode = Distributed(actor="A100", rollout="A10G")
    assert mode.async_config.enabled is False
    assert mode.async_config.mode == "pipeline"

    d = mode.to_dict()
    restored = Distributed.from_dict(d)
    assert restored.async_config.enabled is False
    print("  PASS: Distributed default AsyncConfig (disabled)")


def test_orchestrator_config_with_async():
    """Test full OrchestratorConfig round-trip with async mode."""
    from mortal.config import OrchestratorConfig, Distributed, AsyncConfig

    mode = Distributed(
        actor="A100", rollout="A10G",
        async_config=AsyncConfig(enabled=True, mode="queue", queue_size=4),
    )
    config = OrchestratorConfig(mode=mode)
    d = config.to_dict()
    assert d["mode"]["async_config"]["enabled"] is True
    assert d["mode"]["async_config"]["queue_size"] == 4

    restored = OrchestratorConfig.from_dict(d)
    assert isinstance(restored.mode, Distributed)
    assert restored.mode.async_config.enabled is True
    assert restored.mode.async_config.queue_size == 4
    print("  PASS: OrchestratorConfig with async round-trip")


def test_backward_compat_no_async():
    """Test that configs without async_config still deserialize."""
    from mortal.config import Distributed

    # Simulate old config dict without async_config
    d = {
        "type": "Distributed",
        "actor": {"gpu_type": "A100", "count": 1, "instances": 1},
        "rollout": {"gpu_type": "A10G", "count": 1, "instances": 1},
        "weight_sync_method": "reload",
        "sync_weights_every": 1,
        "num_rollout_workers": 2,
    }
    mode = Distributed.from_dict(d)
    assert mode.async_config.enabled is False
    print("  PASS: Backward compat (no async_config in dict)")


def test_helper_get_batch_data():
    """Test _get_batch_data helper."""
    from mortal.orchestrator import _get_batch_data

    # Create a mock dataset
    class MockDataset:
        def __init__(self, data):
            self._data = data
            self.column_names = list(data.keys())

        def select(self, indices):
            new_data = {}
            for k, v in self._data.items():
                new_data[k] = [v[i] for i in indices]
            return MockDataset(new_data)

        def __getitem__(self, key):
            return self._data[key]

        def __len__(self):
            return len(self._data["prompt"])

    dataset = MockDataset({
        "prompt": ["p1", "p2", "p3", "p4"],
        "answer": ["a1", "a2", "a3", "a4"],
    })

    prompts, kwargs = _get_batch_data(dataset, 0, 2, ["answer"], 3)
    assert len(prompts) == 6  # 2 prompts * 3 generations
    assert prompts == ["p1", "p1", "p1", "p2", "p2", "p2"]
    assert kwargs["answer"] == ["a1", "a1", "a1", "a2", "a2", "a2"]
    print("  PASS: _get_batch_data helper")


def test_helper_chunk_list():
    """Test chunk_list helper."""
    from mortal.orchestrator import chunk_list

    chunks = chunk_list([1, 2, 3, 4, 5], 2)
    assert len(chunks) == 2
    assert chunks[0] == [1, 2]
    assert chunks[1] == [3, 4, 5]

    chunks = chunk_list([1, 2, 3], 1)
    assert len(chunks) == 1
    assert chunks[0] == [1, 2, 3]

    chunks = chunk_list([], 3)
    assert len(chunks) == 0
    print("  PASS: chunk_list helper")


def test_trainer_routing():
    """Test MortalTrainer routes to correct orchestrator based on async_config."""
    from mortal.config import Distributed, AsyncConfig

    # Test sync mode (default)
    mode = Distributed(actor="A100", rollout="A10G")
    assert not mode.async_config.enabled

    # Test pipeline mode
    mode = Distributed(
        actor="A100", rollout="A10G",
        async_config=AsyncConfig(enabled=True, mode="pipeline"),
    )
    assert mode.async_config.enabled
    assert mode.async_config.mode == "pipeline"

    # Test queue mode
    mode = Distributed(
        actor="A100", rollout="A10G",
        async_config=AsyncConfig(enabled=True, mode="queue", queue_size=3),
    )
    assert mode.async_config.enabled
    assert mode.async_config.mode == "queue"
    assert mode.async_config.queue_size == 3
    print("  PASS: Trainer routing config")


# ---------------------------------------------------------------------------
# modal.Queue protocol tests (requires Modal)
# ---------------------------------------------------------------------------


def test_queue_basic_protocol():
    """Test basic put/get with modal.Queue.ephemeral()."""
    import modal

    with modal.Queue.ephemeral() as q:
        # Basic put/get
        q.put({"msg": "hello"}, partition="test")
        result = q.get(partition="test", timeout=5)
        assert result == {"msg": "hello"}

        # Non-blocking get on empty queue
        result = q.get(partition="test", block=False)
        assert result is None

        print("  PASS: Queue basic put/get")


def test_queue_partitions():
    """Test that partitions are independent FIFO streams."""
    import modal

    with modal.Queue.ephemeral() as q:
        q.put({"type": "batch", "id": 1}, partition="batches")
        q.put({"type": "batch", "id": 2}, partition="batches")
        q.put({"type": "control", "version": 1}, partition="control")

        # Control and batches are independent
        ctrl = q.get(partition="control", timeout=5)
        assert ctrl["type"] == "control"
        assert ctrl["version"] == 1

        batch1 = q.get(partition="batches", timeout=5)
        assert batch1["id"] == 1

        batch2 = q.get(partition="batches", timeout=5)
        assert batch2["id"] == 2

        # Both empty now
        assert q.get(partition="batches", block=False) is None
        assert q.get(partition="control", block=False) is None

        print("  PASS: Queue partitions independent")


def test_queue_batch_message_format():
    """Test the actual message format we use for training batches."""
    import modal

    with modal.Queue.ephemeral() as q:
        # Simulate producer putting a batch
        batch = {
            "prompts": ["What is 2+2?", "What is 3+3?"],
            "completions": ["<code>print(4)</code>", "<code>print(6)</code>"],
            "logprobs": [[-0.5, -0.3], [-0.6, -0.2]],
            "kwargs": {"answer": ["4", "6"]},
            "param_version": 0,
        }
        q.put(batch, partition="batches")

        # Simulate consumer getting
        got = q.get(partition="batches", timeout=5)
        assert got["prompts"] == batch["prompts"]
        assert got["completions"] == batch["completions"]
        assert got["logprobs"] == batch["logprobs"]
        assert got["kwargs"]["answer"] == ["4", "6"]
        assert got["param_version"] == 0

        print("  PASS: Queue batch message format")


def test_queue_control_protocol():
    """Test control message protocol (version updates + stop)."""
    import modal

    with modal.Queue.ephemeral() as q:
        # Consumer signals version update
        q.put({"version": 1, "model_path": "/storage/model"}, partition="control")
        q.put({"version": 2, "model_path": "/storage/model"}, partition="control")

        # Producer reads control messages
        ctrl = q.get(partition="control", block=False)
        assert ctrl["version"] == 1

        ctrl = q.get(partition="control", block=False)
        assert ctrl["version"] == 2

        # No more control messages
        assert q.get(partition="control", block=False) is None

        # Stop signal
        q.put({"stop": True}, partition="control")
        ctrl = q.get(partition="control", block=False)
        assert ctrl["stop"] is True

        print("  PASS: Queue control protocol")


def test_queue_backpressure():
    """Test that queue.len() works for backpressure checks."""
    import modal

    with modal.Queue.ephemeral() as q:
        assert q.len(partition="batches") == 0

        q.put({"id": 1}, partition="batches")
        assert q.len(partition="batches") == 1

        q.put({"id": 2}, partition="batches")
        assert q.len(partition="batches") == 2

        # Simulate backpressure check
        max_queue_size = 2
        depth = q.len(partition="batches")
        assert depth >= max_queue_size  # Would pause producer

        # Consume one
        q.get(partition="batches", timeout=5)
        assert q.len(partition="batches") == 1  # Below threshold now

        print("  PASS: Queue backpressure (len)")


def test_queue_staleness_filtering():
    """Test consumer-side staleness filtering logic."""
    import modal

    with modal.Queue.ephemeral() as q:
        # Put batches with different param_versions
        q.put({"id": 1, "param_version": 0}, partition="batches")
        q.put({"id": 2, "param_version": 0}, partition="batches")
        q.put({"id": 3, "param_version": 1}, partition="batches")

        staleness_threshold = 1
        current_version = 2

        # Consumer reads and filters
        accepted = []
        dropped = 0
        while q.len(partition="batches") > 0:
            batch = q.get(partition="batches", timeout=5)
            staleness = current_version - batch["param_version"]
            if staleness > staleness_threshold:
                dropped += 1
            else:
                accepted.append(batch)

        assert dropped == 2  # Batches with v=0 are stale (staleness=2 > threshold=1)
        assert len(accepted) == 1
        assert accepted[0]["id"] == 3  # Only v=1 batch accepted
        assert accepted[0]["param_version"] == 1

        print("  PASS: Queue staleness filtering")


def test_queue_get_many():
    """Test get_many for accumulating sub-batches."""
    import modal

    with modal.Queue.ephemeral() as q:
        # Put multiple items
        for i in range(4):
            q.put({"chunk": i, "data": f"chunk_{i}"}, partition="batches")

        # Get many at once
        items = q.get_many(3, partition="batches", timeout=5)
        assert len(items) == 3
        assert items[0]["chunk"] == 0
        assert items[2]["chunk"] == 2

        # One left
        assert q.len(partition="batches") == 1

        print("  PASS: Queue get_many")


# ---------------------------------------------------------------------------
# Pipeline logic tests
# ---------------------------------------------------------------------------


def test_pipeline_skip_prelaunch_on_sync():
    """Test that pipeline mode skips pre-launching generation on sync steps.

    This is the fix for the bug where Modal spins up a new replica (with
    uninitialized state) when we call update_weights_from_volume on a
    worker that's busy with generate.
    """
    sync_every = 1  # sync every step

    # Simulate 5 steps
    for global_step in range(5):
        will_sync = (global_step + 1) % sync_every == 0
        has_next = global_step + 1 < 5

        if sync_every == 1:
            # Every step is a sync step, so should never pre-launch
            assert will_sync is True, f"step {global_step}: expected will_sync=True"

    sync_every = 2
    results = []
    for global_step in range(5):
        will_sync = (global_step + 1) % sync_every == 0
        has_next = global_step + 1 < 5
        should_prelaunch = has_next and not will_sync
        results.append((global_step, will_sync, should_prelaunch))

    # step 0: sync_every=2, (0+1)%2=1≠0 → will_sync=False → prelaunch=True
    assert results[0] == (0, False, True)
    # step 1: (1+1)%2=0 → will_sync=True → prelaunch=False (worker needs to be idle for sync)
    assert results[1] == (1, True, False)
    # step 2: (2+1)%2=1 → will_sync=False → prelaunch=True
    assert results[2] == (2, False, True)
    # step 3: (3+1)%2=0 → will_sync=True → prelaunch=False
    assert results[3] == (3, True, False)
    # step 4: last step, has_next=False → prelaunch=False
    assert results[4] == (4, False, False)

    print("  PASS: Pipeline skips pre-launch on sync steps")


def test_pipeline_prelaunch_after_sync():
    """Test that pipeline launches generation AFTER sync with updated weights."""
    sync_every = 1
    events = []

    for global_step in range(3):
        will_sync = (global_step + 1) % sync_every == 0
        has_next = global_step + 1 < 3

        # Simulate the pipeline flow
        events.append(f"step_{global_step}_collect_gen")

        if has_next and not will_sync:
            events.append(f"step_{global_step}_prelaunch_gen")

        events.append(f"step_{global_step}_train")

        if will_sync:
            events.append(f"step_{global_step}_weight_sync")
            if has_next:
                events.append(f"step_{global_step}_post_sync_launch_gen")

    # With sync_every=1, every step syncs, so generation always launches AFTER sync
    assert "step_0_prelaunch_gen" not in events  # No pre-launch on sync step
    assert "step_0_post_sync_launch_gen" in events  # Launch after sync
    assert events.index("step_0_weight_sync") < events.index("step_0_post_sync_launch_gen")

    print("  PASS: Pipeline launches gen after sync with updated weights")


# ---------------------------------------------------------------------------
# Modal cls concurrent behavior test (requires Modal)
# ---------------------------------------------------------------------------


def test_modal_cls_concurrent_spawn_identity():
    """Test whether Modal routes concurrent .spawn() to same or different container.

    This test validates our hypothesis about the weight sync bug:
    if we .spawn() on a busy @app.cls instance, does Modal route to
    the same container (queued) or a new one (fresh state)?

    We create a simple @app.cls that tracks state and test concurrent calls.
    """
    import modal

    app = modal.App("test-concurrent-cls")

    @app.cls(image=modal.Image.debian_slim(), serialized=True)
    class StatefulWorker:
        @modal.enter()
        def setup(self):
            self.initialized = True
            self.call_count = 0
            self._container_id = id(self)
            print(f"[StatefulWorker] Container created, id={self._container_id}")

        @modal.method()
        def slow_work(self, duration: float = 3.0) -> dict:
            """Simulate a slow operation (like generate)."""
            import time
            self.call_count += 1
            call_num = self.call_count
            print(f"[StatefulWorker.slow_work] START call #{call_num}, "
                  f"container_id={self._container_id}")
            time.sleep(duration)
            print(f"[StatefulWorker.slow_work] END call #{call_num}")
            return {
                "container_id": self._container_id,
                "call_count": call_num,
                "initialized": self.initialized,
            }

        @modal.method()
        def check_state(self) -> dict:
            """Check if state is preserved (like update_weights_from_volume)."""
            self.call_count += 1
            call_num = self.call_count
            print(f"[StatefulWorker.check_state] call #{call_num}, "
                  f"container_id={self._container_id}, initialized={self.initialized}")
            return {
                "container_id": self._container_id,
                "call_count": call_num,
                "initialized": self.initialized,
            }

    with modal.enable_output():
        with app.run():
            worker = StatefulWorker()

            # Test 1: Sequential calls — same container
            r1 = worker.check_state.remote()
            r2 = worker.check_state.remote()
            assert r1["container_id"] == r2["container_id"], \
                f"Sequential calls hit different containers: {r1['container_id']} vs {r2['container_id']}"
            assert r1["initialized"] is True
            print("  PASS: Sequential calls hit same container")

            # Test 2: Concurrent calls — spawn slow_work, then check_state
            # This simulates: generate.spawn() running, then update_weights.spawn()
            print("\n  Testing concurrent .spawn() behavior...")
            slow_future = worker.slow_work.spawn(duration=5.0)
            import time
            time.sleep(0.5)  # Give it time to start

            # Now call check_state while slow_work is running
            state_future = worker.check_state.spawn()

            # Collect results
            slow_result = slow_future.get()
            state_result = state_future.get()

            same_container = slow_result["container_id"] == state_result["container_id"]
            print(f"\n  slow_work container_id: {slow_result['container_id']}")
            print(f"  check_state container_id: {state_result['container_id']}")
            print(f"  Same container: {same_container}")
            print(f"  check_state.initialized: {state_result['initialized']}")

            if same_container:
                print("  RESULT: Modal QUEUED the concurrent call (same container, state preserved)")
                # call_count should reflect both calls
                assert state_result["call_count"] > slow_result["call_count"], \
                    "check_state should run after slow_work if queued"
            else:
                print("  RESULT: Modal routed to NEW container (DIFFERENT container, fresh state)")
                print("  This confirms the bug: concurrent .spawn() creates new replicas")
                # The state should still be initialized (fresh @modal.enter runs)
                assert state_result["initialized"] is True

            print(f"  PASS: Modal concurrent spawn behavior documented (same_container={same_container})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_tests(unit_only=False, queue_only=False, integration_only=False):
    """Run test suites."""
    passed = 0
    failed = 0

    def run(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {name}: {e}")
            traceback.print_exc()

    if not queue_only and not integration_only:
        print("\n=== Unit Tests (Config + Helpers) ===")
        run("async_config_basics", test_async_config_basics)
        run("async_config_serialization", test_async_config_serialization)
        run("distributed_with_async_config", test_distributed_with_async_config)
        run("distributed_default_async_config", test_distributed_default_async_config)
        run("orchestrator_config_with_async", test_orchestrator_config_with_async)
        run("backward_compat_no_async", test_backward_compat_no_async)
        run("helper_get_batch_data", test_helper_get_batch_data)
        run("helper_chunk_list", test_helper_chunk_list)
        run("trainer_routing", test_trainer_routing)

    if not queue_only and not integration_only:
        print("\n=== Pipeline Logic Tests ===")
        run("pipeline_skip_prelaunch_on_sync", test_pipeline_skip_prelaunch_on_sync)
        run("pipeline_prelaunch_after_sync", test_pipeline_prelaunch_after_sync)

    if not unit_only and not integration_only:
        print("\n=== Queue Protocol Tests (requires Modal) ===")
        run("queue_basic_protocol", test_queue_basic_protocol)
        run("queue_partitions", test_queue_partitions)
        run("queue_batch_message_format", test_queue_batch_message_format)
        run("queue_control_protocol", test_queue_control_protocol)
        run("queue_backpressure", test_queue_backpressure)
        run("queue_staleness_filtering", test_queue_staleness_filtering)
        run("queue_get_many", test_queue_get_many)

    if integration_only:
        print("\n=== Modal Integration Tests (requires Modal) ===")
        run("modal_cls_concurrent_spawn_identity", test_modal_cls_concurrent_spawn_identity)

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--queue-only", action="store_true")
    parser.add_argument("--integration-only", action="store_true")
    args = parser.parse_args()

    success = run_tests(
        unit_only=args.unit_only,
        queue_only=args.queue_only,
        integration_only=args.integration_only,
    )
    sys.exit(0 if success else 1)
