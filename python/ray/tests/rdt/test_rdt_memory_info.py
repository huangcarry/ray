"""Tests for the experimental per-actor RDT memory snapshot API."""
import gc
import sys
import threading
import weakref

import pytest
import torch

import ray
from ray.experimental.rdt.rdt_store import (
    RDTMemoryInfo,
    RDTStore,
    _inspect_tensor,
    _union_span_bytes,
    _validate_memory_info_timeout,
    get_memory_info,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot(store, actor_id="a", node_id="n", pid=1):
    return store.get_memory_info_snapshot(actor_id, node_id, pid)


def _row(info, device):
    for row in info.devices:
        if row.device == device:
            return row
    return None


class _BrokenStorageTensor(torch.Tensor):
    """A torch.Tensor subclass whose storage reads fail, for testing the
    controlled ``failed_storage_read`` reason."""

    def untyped_storage(self):
        raise RuntimeError("broken storage")


# ---------------------------------------------------------------------------
# Pure unit tests (no Ray cluster)
# ---------------------------------------------------------------------------


def test_empty_store_returns_zero_counts():
    info = _snapshot(RDTStore())
    assert isinstance(info, RDTMemoryInfo)
    assert info.devices == []
    assert info.num_error_entries == 0
    assert info.num_uninspectable_tensors == 0
    assert info.incomplete is False
    assert info.unsupported == []


def test_union_span_bytes():
    assert _union_span_bytes([]) == 0
    assert _union_span_bytes([(0, 0)]) == 0
    assert _union_span_bytes([(0, 10)]) == 10
    assert _union_span_bytes([(0, 10), (5, 15)]) == 15  # overlapping
    assert _union_span_bytes([(0, 10), (10, 20)]) == 20  # adjacent
    assert _union_span_bytes([(0, 10), (20, 30)]) == 20  # disjoint
    assert _union_span_bytes([(5, 15), (0, 10), (20, 25), (22, 30)]) == 25


def test_single_tensor_counts():
    store = RDTStore()
    tensor = torch.empty(16, dtype=torch.float32)
    store.add_object("obj1", [tensor])
    info = _snapshot(store)
    row = _row(info, "cpu")
    assert row is not None
    assert row.num_objects == 1
    assert row.num_copies == 1
    assert row.tensor_count == 1
    assert row.tensor_bytes == 16 * 4
    assert row.referenced_storage_bytes == tensor.untyped_storage().nbytes()


def test_shared_views_deduplicate_storage():
    store = RDTStore()
    base = torch.empty(100, dtype=torch.float32)
    a = base[:50]
    b = base[50:]
    store.add_object("obj_views", [a, b])
    info = _snapshot(store)
    row = _row(info, "cpu")
    assert row.num_objects == 1
    assert row.num_copies == 1
    assert row.tensor_count == 2
    assert row.tensor_bytes == 100 * 4  # 50 * 4 + 50 * 4
    # Both views reference the same backing storage.
    assert row.referenced_storage_bytes == 100 * 4


def test_overlapping_external_buffers_deduplicate():
    # Two distinct storage objects (from two frombuffer calls) that alias the
    # same external bytes must not double-count referenced storage.
    store = RDTStore()
    buf = bytearray(100 * 4)
    t1 = torch.frombuffer(buf, dtype=torch.float32)
    t2 = torch.frombuffer(buf, dtype=torch.float32)
    assert t1.untyped_storage() is not t2.untyped_storage()
    assert t1.untyped_storage().data_ptr() == t2.untyped_storage().data_ptr()
    store.add_object("obj_fb1", [t1])
    store.add_object("obj_fb2", [t2])
    info = _snapshot(store)
    row = _row(info, "cpu")
    assert row.num_objects == 2
    assert row.tensor_count == 2
    assert row.tensor_bytes == 2 * 100 * 4
    assert row.referenced_storage_bytes == 100 * 4


def test_repeated_tensor_across_objects():
    store = RDTStore()
    tensor = torch.empty(8, dtype=torch.float32)
    store.add_object("obj1", [tensor])
    store.add_object("obj2", [tensor])
    info = _snapshot(store)
    row = _row(info, "cpu")
    assert row.num_objects == 2
    assert row.num_copies == 2
    assert row.tensor_count == 2
    assert row.tensor_bytes == 2 * 8 * 4
    assert row.referenced_storage_bytes == tensor.untyped_storage().nbytes()


def test_multiple_copies_same_object_id():
    store = RDTStore()
    tensor = torch.empty(4, dtype=torch.float32)
    store.add_object("obj1", [tensor])
    store.add_object("obj1", [tensor])
    info = _snapshot(store)
    row = _row(info, "cpu")
    assert row.num_objects == 1  # one unique ID
    assert row.num_copies == 2  # two queued entries
    assert row.tensor_count == 2
    assert row.tensor_bytes == 2 * 4 * 4
    assert row.referenced_storage_bytes == tensor.untyped_storage().nbytes()


def test_zero_size_view_references_backing_storage():
    store = RDTStore()
    base = torch.empty(10, dtype=torch.float32)
    view = base[0:0]  # zero numel, nonempty backing storage
    store.add_object("obj_zero", [view])
    info = _snapshot(store)
    row = _row(info, "cpu")
    assert row.tensor_count == 1
    assert row.tensor_bytes == 0
    assert row.referenced_storage_bytes == base.untyped_storage().nbytes()


def test_empty_tensor_has_zero_storage_bytes():
    store = RDTStore()
    empty = torch.empty(0, dtype=torch.float32)
    store.add_object("obj_empty", [empty])
    info = _snapshot(store)
    row = _row(info, "cpu")
    assert row.tensor_count == 1
    assert row.tensor_bytes == 0
    assert row.referenced_storage_bytes == 0


def test_error_entries_counted_separately():
    store = RDTStore()
    store.add_object("obj_err", RuntimeError("recv failed"))
    info = _snapshot(store)
    assert info.num_error_entries == 1
    assert info.devices == []
    assert info.incomplete is False


def test_custom_payload_is_incomplete_not_zero():
    store = RDTStore()
    store.add_object("obj_custom", [object()])
    info = _snapshot(store)
    assert info.incomplete is True
    assert info.num_uninspectable_tensors == 1
    assert info.unsupported == ["unsupported_object_type:object"]
    assert info.devices == []


def test_duplicate_unsupported_reasons_are_deduplicated():
    store = RDTStore()
    store.add_object("obj_custom", [object(), object()])
    info = _snapshot(store)
    assert info.incomplete is True
    assert info.num_uninspectable_tensors == 2
    assert info.unsupported == ["unsupported_object_type:object"]


def test_sparse_layout_is_unsupported():
    store = RDTStore()
    sparse = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1]]), values=torch.tensor([1.0, 2.0]), size=(2,)
    )
    store.add_object("obj_sparse", [sparse])
    info = _snapshot(store)
    assert info.incomplete is True
    assert info.num_uninspectable_tensors == 1
    assert "unsupported_tensor_layout:torch.sparse_coo" in info.unsupported


def test_meta_tensor_is_unsupported():
    store = RDTStore()
    meta = torch.empty(4, device="meta")
    store.add_object("obj_meta", [meta])
    info = _snapshot(store)
    assert info.incomplete is True
    assert info.num_uninspectable_tensors == 1
    assert "unsupported_tensor_device:meta" in info.unsupported


def test_storage_read_failure_is_incomplete():
    store = RDTStore()
    broken = torch.empty(4, dtype=torch.float32).as_subclass(_BrokenStorageTensor)
    store.add_object("obj_broken", [broken])
    info = _snapshot(store)
    assert info.incomplete is True
    assert info.num_uninspectable_tensors == 1
    assert info.unsupported == ["failed_storage_read"]


def test_mixed_supported_and_unsupported_entries():
    store = RDTStore()
    tensor = torch.empty(6, dtype=torch.float32)
    store.add_object("obj_ok", [tensor])
    store.add_object("obj_bad", [object()])
    info = _snapshot(store)
    assert info.incomplete is True
    assert info.num_uninspectable_tensors == 1
    assert info.unsupported == ["unsupported_object_type:object"]
    row = _row(info, "cpu")
    # The valid entry remains queryable.
    assert row.num_objects == 1
    assert row.tensor_count == 1
    assert row.tensor_bytes == 6 * 4


def test_snapshot_holds_no_tensor_references():
    store = RDTStore()
    tensor = torch.empty(10, dtype=torch.float32)
    expected = tensor.untyped_storage().nbytes()
    ref = weakref.ref(tensor)
    store.add_object("obj1", [tensor])
    # Keep the returned snapshot alive through the GC to prove it retains no
    # strong reference to the tensor.
    snapshot = _snapshot(store)
    store.pop_object("obj1")
    del tensor
    gc.collect()
    assert ref() is None
    assert snapshot.devices[0].referenced_storage_bytes == expected


def test_concurrent_mutation_is_safe():
    store = RDTStore()
    stop = threading.Event()
    errors = []

    def writer():
        try:
            i = 0
            while not stop.is_set():
                store.add_object(f"obj_{i}", [torch.empty(8, dtype=torch.float32)])
                i += 1
        except Exception as e:  # pragma: no cover - only on failure
            errors.append(e)

    def popper():
        try:
            while not stop.is_set():
                with store._lock:
                    obj_id = next(iter(store._rdt_store)) if store._rdt_store else None
                if obj_id is not None:
                    store.pop_object(obj_id)
        except Exception as e:  # pragma: no cover - only on failure
            errors.append(e)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=popper),
    ]
    for t in threads:
        t.start()
    try:
        for _ in range(300):
            info = _snapshot(store)
            assert isinstance(info, RDTMemoryInfo)
            for row in info.devices:
                assert row.num_objects >= 0
                assert row.num_copies >= row.num_objects
                assert row.tensor_count >= row.num_copies
                assert row.tensor_bytes >= 0
                assert row.referenced_storage_bytes >= 0
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10)
    assert errors == []


def test_inspect_tensor_returns_reason_for_non_tensor():
    accounting, reason = _inspect_tensor(object(), torch)
    assert accounting is None
    assert reason == "unsupported_object_type:object"


def test_validate_timeout():
    _validate_memory_info_timeout(0.1)
    _validate_memory_info_timeout(10.0)
    for bad in [0, -1, 0.0, -0.5]:
        with pytest.raises(ValueError):
            _validate_memory_info_timeout(bad)
    for bad in [float("inf"), float("nan"), None, "10", True, 10**1000]:
        with pytest.raises(ValueError):
            _validate_memory_info_timeout(bad)
    with pytest.raises(ValueError):
        _validate_memory_info_timeout(1e9)


# ---------------------------------------------------------------------------
# Actor integration tests
# ---------------------------------------------------------------------------


@ray.remote(enable_tensor_transport=True)
class MemoryInfoActor:
    def seed_shared_views(self):
        from ray._private.worker import global_worker

        store = global_worker.rdt_manager.rdt_store
        base = torch.empty(100, dtype=torch.float32)
        a = base[:50]
        b = base[50:]
        store.add_object("obj_shared", [a, b], is_primary=True)
        return base.untyped_storage().nbytes()

    def seed_error(self):
        from ray._private.worker import global_worker

        store = global_worker.rdt_manager.rdt_store
        store.add_object("obj_error", RuntimeError("recv failed"))

    def rdt_state_untouched(self):
        from ray._private.worker import global_worker

        manager = global_worker._rdt_manager
        return manager is None or manager._rdt_store is None

    def my_actor_id(self):
        return ray.get_runtime_context().get_actor_id()


@ray.remote(enable_tensor_transport=True)
class BlockingActor:
    def __init__(self):
        self._started = False

    @ray.method(concurrency_group="_ray_system")
    def block_background(self):
        self._started = True
        import time

        time.sleep(30)

    def started(self):
        return self._started


@ray.remote(enable_tensor_transport=True)
class GlooLifecycleActor:
    @ray.method(tensor_transport="gloo")
    def echo(self, tensor):
        return tensor

    def get_num_objects(self):
        from ray._private.worker import global_worker

        return global_worker.rdt_manager.rdt_store.get_num_objects()


@ray.remote
class PlainActor:
    pass


def test_get_memory_info_untouched_actor(ray_start_regular):
    actor = MemoryInfoActor.remote()
    info = get_memory_info(actor)
    assert isinstance(info, RDTMemoryInfo)
    assert info.devices == []
    assert info.num_error_entries == 0
    assert info.num_uninspectable_tensors == 0
    assert info.incomplete is False
    assert info.actor_id
    assert info.node_id
    assert info.pid > 0
    assert info.actor_id == ray.get(actor.my_actor_id.remote())
    # Querying an untouched actor must not create the RDT manager or store.
    assert ray.get(actor.rdt_state_untouched.remote()) is True


def test_get_memory_info_seeded_store(ray_start_regular):
    actor = MemoryInfoActor.remote()
    expected_storage = ray.get(actor.seed_shared_views.remote())
    info = get_memory_info(actor)
    row = _row(info, "cpu")
    assert row is not None
    assert row.num_objects == 1
    assert row.num_copies == 1
    assert row.tensor_count == 2
    assert row.tensor_bytes == 100 * 4
    assert row.referenced_storage_bytes == expected_storage


def test_get_memory_info_reports_error_entries(ray_start_regular):
    actor = MemoryInfoActor.remote()
    ray.get(actor.seed_error.remote())
    info = get_memory_info(actor)
    assert info.num_error_entries == 1
    assert info.devices == []


def test_get_memory_info_gloo_lifecycle(ray_start_regular):
    """Validate the source-actor production lifecycle with the GLOO transport.

    This checks that a real CPU RDT object created by an actor task annotated
    with ``@ray.method(tensor_transport="gloo")`` is visible in the source
    actor's store, and that its primary copy is released once the ObjectRef is
    freed. It exercises object creation and release, not a transfer to a
    second actor.
    """
    from ray._common.test_utils import wait_for_condition
    from ray.experimental.collective import create_collective_group

    actors = [GlooLifecycleActor.remote() for _ in range(2)]
    create_collective_group(actors, backend="gloo")
    sender = actors[0]

    tensor = torch.randn(32, dtype=torch.float32)
    ref = sender.echo.remote(tensor)

    # Wait for the RDT object to be created in the sender's store.
    wait_for_condition(
        lambda: ray.get(sender.get_num_objects.remote()) == 1, timeout=10
    )
    info = get_memory_info(sender)
    row = _row(info, "cpu")
    assert row is not None
    assert row.num_objects == 1
    assert row.tensor_count == 1
    assert row.tensor_bytes == 32 * 4
    assert row.referenced_storage_bytes == tensor.untyped_storage().nbytes()

    del ref
    # Freeing the ObjectRef releases the sender's primary copy.
    wait_for_condition(
        lambda: ray.get(sender.get_num_objects.remote()) == 0, timeout=10
    )
    info = get_memory_info(sender)
    assert info.devices == []
    assert info.num_error_entries == 0


def test_get_memory_info_requires_tensor_transport(ray_start_regular):
    actor = PlainActor.remote()
    with pytest.raises(ValueError, match="enable_tensor_transport=True"):
        get_memory_info(actor)


def test_get_memory_info_rejects_non_actor(ray_start_regular):
    with pytest.raises(TypeError, match="actor handle"):
        get_memory_info(object())


@pytest.mark.parametrize(
    "bad_timeout", [0, -1, float("inf"), float("nan"), None, 10**1000]
)
def test_get_memory_info_invalid_timeout(ray_start_regular, bad_timeout):
    actor = MemoryInfoActor.remote()
    with pytest.raises(ValueError):
        get_memory_info(actor, timeout=bad_timeout)


def test_get_memory_info_timeout(ray_start_regular):
    from ray._common.test_utils import wait_for_condition

    actor = BlockingActor.remote()
    block_ref = actor.block_background.remote()
    # Confirm the blocking task has actually started on the single-threaded
    # _ray_system concurrency group before submitting the diagnostic.
    wait_for_condition(lambda: ray.get(actor.started.remote()), timeout=10)
    with pytest.raises(ray.exceptions.GetTimeoutError):
        get_memory_info(actor, timeout=0.5)
    # Keep the blocking task's ref alive until the assertion completes.
    assert block_ref is not None


def test_get_memory_info_dead_actor(ray_start_regular):
    actor = MemoryInfoActor.remote()
    ray.kill(actor)
    with pytest.raises(ray.exceptions.RayActorError):
        get_memory_info(actor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_get_memory_info_multiple_devices_store():
    store = RDTStore()
    cpu_t = torch.empty(10, dtype=torch.float32)
    cuda_t = torch.empty(20, dtype=torch.float32, device="cuda")
    store.add_object("obj_cpu", [cpu_t])
    store.add_object("obj_cuda", [cuda_t])
    info = _snapshot(store)
    by_device = {d.device: d for d in info.devices}
    assert "cpu" in by_device
    assert "cuda:0" in by_device
    assert by_device["cpu"].num_objects == 1
    assert by_device["cpu"].tensor_bytes == 10 * 4
    assert by_device["cuda:0"].num_objects == 1
    assert by_device["cuda:0"].tensor_bytes == 20 * 4
    assert (
        by_device["cuda:0"].referenced_storage_bytes
        == cuda_t.untyped_storage().nbytes()
    )


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
