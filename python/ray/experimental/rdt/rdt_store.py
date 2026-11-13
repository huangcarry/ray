import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from ray.experimental.rdt.tensor_transport_manager import (
    CommunicatorMetadata,
    TensorTransportMetadata,
)
from ray.experimental.rdt.util import (
    device_match_transport,
    get_tensor_transport_manager,
)
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)

# Upper bound on the public `get_memory_info` timeout. The timeout only bounds
# the caller's `ray.get` wait; it does not cancel a diagnostic already queued
# or running on the actor.
_MAX_MEMORY_INFO_TIMEOUT_SECONDS = 3600.0


@PublicAPI(stability="alpha")
@dataclass
class RDTDeviceMemoryInfo:
    """Per-device memory accounting for a single process-local device.

    Device keys are process-local (``"cpu"``, ``"cuda:0"``), not global GPU
    identities. No cross-process or cross-actor deduplication is claimed.

    Args:
        device: The process-local device key.
        num_objects: Number of unique RDT object IDs that have at least one
            non-error entry touching this device.
        num_copies: Number of queued non-error RDT object entries (copies)
            touching this device.
        tensor_count: Number of logical tensor occurrences on this device.
        tensor_bytes: Sum of ``numel * element_size`` over every tensor
            occurrence on this device.
        referenced_storage_bytes: Total bytes of tensor backing storage
            referenced on this device, deduplicated over overlapping byte spans.
    """

    device: str
    num_objects: int
    num_copies: int
    tensor_count: int
    tensor_bytes: int
    referenced_storage_bytes: int


@PublicAPI(stability="alpha")
@dataclass
class RDTMemoryInfo:
    """A snapshot of one actor's RDT store memory accounting.

    This is referenced tensor backing storage, not allocated or cached CUDA
    memory, registered bytes, NIXL pool utilization, or free/available GPU
    memory.

    Args:
        actor_id: The actor's ID.
        node_id: The node ID of the process hosting the actor.
        pid: The OS process ID of the actor's worker.
        devices: Per-device accounting, sorted by device key.
        num_error_entries: Number of failed/error entries in the store (for
            example, a failed receive). These are never attributed to a device.
        num_uninspectable_tensors: Number of stored tensors or payloads that
            could not be inspected, such as custom payloads or unsupported
            tensor layouts or devices. These are never guessed as zero.
        incomplete: Whether any stored entry or tensor could not be fully
            accounted for. When True, the numeric fields cover only the entries
            that could be inspected.
        unsupported: Deduplicated, stable categories describing why entries
            could not be accounted for when ``incomplete`` is True. Categories
            never include raw tensor addresses.
    """

    actor_id: str
    node_id: str
    pid: int
    devices: List[RDTDeviceMemoryInfo]
    num_error_entries: int
    num_uninspectable_tensors: int
    incomplete: bool
    unsupported: List[str]


def __ray_send__(
    self,
    obj_id: str,
    tensor_transport_meta: TensorTransportMetadata,
    communicator_meta: CommunicatorMetadata,
    backend: str,
):
    """Helper function that runs on the src actor to send tensors to the dst actor."""
    from ray._private.worker import global_worker

    rdt_store = global_worker.rdt_manager._rdt_store
    assert rdt_store.has_object(obj_id), f"obj_id={obj_id} not found in RDT store"

    tensors = rdt_store.get_object(obj_id)

    tensor_transport_manager = get_tensor_transport_manager(backend)
    tensor_transport_manager.send_multiple_tensors(
        tensors,
        tensor_transport_meta,
        communicator_meta,
    )


def validate_tensor_buffers(
    tensor_buffers: List["torch.Tensor"],
    tensor_meta: List[Tuple["torch.Size", "torch.dtype"]],
):
    if len(tensor_buffers) != len(tensor_meta):
        raise ValueError(
            f"Length of tensor_buffers ({len(tensor_buffers)}) does not match length from object metadata ({len(tensor_meta)})."
        )

    def tensor_buffer_mismatch_msg(prop, idx, actual, expected):
        return f"{prop} of tensor_buffer at index {idx} ({actual}) does not match {prop.lower()} from object metadata ({expected})."

    for idx, single_buffer in enumerate(tensor_buffers):
        shape, dtype = tensor_meta[idx]
        if single_buffer.shape != shape:
            raise ValueError(
                tensor_buffer_mismatch_msg("Shape", idx, single_buffer.shape, shape)
            )
        if single_buffer.dtype != dtype:
            raise ValueError(
                tensor_buffer_mismatch_msg("Dtype", idx, single_buffer.dtype, dtype)
            )
        if not single_buffer.is_contiguous():
            raise ValueError(f"Tensor buffer at index {idx} is not contiguous.")


def __ray_recv__(
    self,
    obj_id: str,
    tensor_transport_meta: TensorTransportMetadata,
    communicator_meta: CommunicatorMetadata,
    backend: str,
):
    """Helper function that runs on the dst actor to receive tensors from the src actor."""
    from ray._private.worker import global_worker

    rdt_store = global_worker.rdt_manager.rdt_store
    try:
        tensor_transport_manager = get_tensor_transport_manager(backend)
        tensors = tensor_transport_manager.recv_multiple_tensors(
            obj_id,
            tensor_transport_meta,
            communicator_meta,
        )
        assert len(tensors) == len(tensor_transport_meta.tensor_meta)
        rdt_store.add_object(obj_id, tensors)
    except Exception as e:
        # Store the error as an RDT object if the recv fails, so waiters will raise the error.
        rdt_store.add_object(obj_id, e)


def __ray_abort_transport__(
    self, obj_id: str, communicator_meta: CommunicatorMetadata, backend: str
):
    """Helper function that can run on an actor doing a send or recv to abort the transport."""
    tensor_transport_manager = get_tensor_transport_manager(backend)
    tensor_transport_manager.abort_transport(obj_id, communicator_meta)


def __ray_free__(
    self,
    obj_id: str,
    tensor_transport_backend: str,
    tensor_transport_meta: TensorTransportMetadata,
):
    try:
        from ray._private.worker import global_worker

        rdt_manager = global_worker.rdt_manager
        rdt_store = rdt_manager.rdt_store

        if not rdt_store.has_object(obj_id):
            return
        tensors = rdt_store.get_object(obj_id)
        try:
            tensor_transport_manager = get_tensor_transport_manager(
                tensor_transport_backend
            )
            tensor_transport_manager.garbage_collect(
                obj_id, tensor_transport_meta, tensors
            )
        except Exception:
            logger.exception(
                "Failed to garbage collect RDT object %s with transport %s.",
                obj_id,
                tensor_transport_backend,
            )
        finally:
            rdt_store.pop_object(obj_id)
    except AssertionError:
        # This could fail if this is a retry and it's already been freed.
        pass


def __ray_fetch_rdt_object__(self, obj_id: str):
    """Helper function that runs on the src actor to fetch tensors from the RDT store via the object store."""
    from ray._private.worker import global_worker

    rdt_store = global_worker.rdt_manager.rdt_store
    rdt_object = rdt_store.wait_and_get_object(obj_id)
    return rdt_object


def _validate_memory_info_timeout(timeout: float) -> None:
    """Validate the public `get_memory_info` timeout argument."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError(f"timeout must be a positive finite number, got {timeout!r}.")
    # Check the upper bound before math.isfinite: converting a huge integer to
    # float inside math.isfinite would raise OverflowError.
    if timeout > _MAX_MEMORY_INFO_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout must be at most {_MAX_MEMORY_INFO_TIMEOUT_SECONDS} seconds, "
            f"got {timeout!r}."
        )
    import math

    if not math.isfinite(timeout):
        raise ValueError(f"timeout must be finite, got {timeout!r}.")
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}.")


def _inspect_tensor(
    tensor: Any, torch: Optional[Any]
) -> Tuple[Optional[Tuple[str, int, int, int]], Optional[str]]:
    """Inspect one stored tensor for memory accounting.

    Returns ``((device, span_start, span_end, tensor_bytes), None)`` for a
    supported torch strided CPU/CUDA tensor, where the byte span covers the
    tensor's whole backing storage. Returns ``(None, category)`` when the tensor
    cannot be accounted for, where ``category`` is a stable, deduplicatable
    string that never embeds raw addresses or arbitrary payload state. Never
    invokes ``__sizeof__`` on arbitrary payloads.
    """
    if torch is None or not isinstance(tensor, torch.Tensor):
        return None, f"unsupported_object_type:{type(tensor).__name__}"
    try:
        if tensor.layout != torch.strided:
            return None, f"unsupported_tensor_layout:{tensor.layout}"
        if tensor.device.type not in ("cpu", "cuda"):
            return None, f"unsupported_tensor_device:{tensor.device.type}"
        storage = tensor.untyped_storage()
        start = storage.data_ptr()
        nbytes = storage.nbytes()
        tensor_bytes = tensor.numel() * tensor.element_size()
    except Exception:
        # Custom tensor subclasses may raise on any of the above accesses; keep
        # the supported entries queryable and report a controlled, address-free
        # reason rather than propagating.
        return None, "failed_storage_read"
    return (str(tensor.device), start, start + nbytes, tensor_bytes), None


def _union_span_bytes(spans: List[Tuple[int, int]]) -> int:
    """Return the total bytes covered by the union of ``[start, end)`` spans.

    Zero-length spans are dropped so empty storages never alias unrelated
    objects.
    """
    spans = [(start, end) for start, end in spans if end > start]
    if not spans:
        return 0
    spans.sort()
    total = 0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def __ray_get_memory_info__(self) -> RDTMemoryInfo:
    """Helper that runs on the actor to build an RDT memory snapshot.

    Reads the worker's backing ``_rdt_manager`` and ``_rdt_store`` fields
    directly rather than through their lazily-creating properties, so querying
    an actor that has never used RDT returns a zero snapshot without creating
    an ``RDTStore`` or importing torch.
    """
    import os

    import ray
    from ray._private.worker import global_worker

    runtime_context = ray.get_runtime_context()
    actor_id = runtime_context.get_actor_id() or ""
    node_id = runtime_context.get_node_id() or ""
    pid = os.getpid()

    rdt_manager = global_worker._rdt_manager
    if rdt_manager is None or rdt_manager._rdt_store is None:
        return RDTMemoryInfo(
            actor_id=actor_id,
            node_id=node_id,
            pid=pid,
            devices=[],
            num_error_entries=0,
            num_uninspectable_tensors=0,
            incomplete=False,
            unsupported=[],
        )
    return rdt_manager._rdt_store.get_memory_info_snapshot(actor_id, node_id, pid)


@PublicAPI(stability="alpha")
def get_memory_info(actor: Any, *, timeout: float = 10.0) -> RDTMemoryInfo:
    """Return a snapshot of one actor's RDT store memory accounting.

    This is an opt-in, synchronous diagnostic query of the provided actor
    handle. It runs on the actor in the ``_ray_system`` concurrency group and
    returns per-device object counts, logical tensor bytes, and deduplicated
    referenced backing-storage bytes. It is not an automatic cluster scan and
    does not integrate with the ``ray memory`` CLI.

    The returned bytes describe referenced tensor backing storage, not
    allocated or cached CUDA memory, registered bytes, NIXL pool utilization,
    or free/available GPU memory. A view into an allocator pool references the
    whole pool, so logical ``tensor_bytes`` may exceed
    ``referenced_storage_bytes`` due to repeated or overlapping views.

    Args:
        actor: The actor to query. Must be created with
            ``@ray.remote(enable_tensor_transport=True)`` so the ``_ray_system``
            concurrency group exists.
        timeout: Maximum time in seconds to wait for the result. Must be a
            positive, finite number no greater than
            ``_MAX_MEMORY_INFO_TIMEOUT_SECONDS``. The timeout only bounds the
            caller's ``ray.get`` wait; it does not cancel a diagnostic already
            queued or running on the actor.

    Returns:
        An :class:`RDTMemoryInfo` snapshot.

    Raises:
        TypeError: If ``actor`` is not an actor handle.
        ValueError: If the actor does not have tensor transport enabled, or if
            ``timeout`` is invalid.
    """
    import ray
    from ray.actor import ActorHandle

    if not isinstance(actor, ActorHandle):
        raise TypeError(
            "get_memory_info expects an actor handle, got "
            f"{type(actor).__name__}. Pass an actor created with "
            "@ray.remote(enable_tensor_transport=True)."
        )
    if not getattr(actor, "_ray_enable_tensor_transport", False):
        raise ValueError(
            "get_memory_info requires an actor created with "
            "@ray.remote(enable_tensor_transport=True). The diagnostic runs in "
            "the actor's _ray_system concurrency group, which exists only for "
            "tensor-transport-enabled actors."
        )
    _validate_memory_info_timeout(timeout)

    ref = actor.__ray_call__.options(concurrency_group="_ray_system").remote(
        __ray_get_memory_info__
    )
    return ray.get(ref, timeout=timeout)


@dataclass
class _RDTObject:
    # A list of tensors representing the RDT object.
    data: List[Any]
    # Whether the RDT object is the primary copy.
    is_primary: bool
    # If a recv failed, we store the error here.
    error: Optional[Exception] = None


class RDTStore:
    """This class is thread-safe. The GPU object store is meant to be read and
    written by the following threads:
    1. The main thread, which is executing user code. This thread may get, put,
    and pop objects.
    2. The background _ray_system thread, which executes data transfers. This
    thread may get and put objects.
    3. The background CoreWorker server thread, which executes garbage
    collection callbacks that pop objects that are no longer in use.
    """

    def __init__(self):
        # A dictionary that maps from an object ID to a queue of tensor lists.
        #
        # Note: Currently, `_rdt_store` is only supported for Ray Actors.
        self._rdt_store: Dict[str, deque[_RDTObject]] = defaultdict(deque)
        # Mapping from tensor data pointer to the IDs of objects that contain it.
        self._tensor_to_object_ids: Dict[int, Set[str]] = defaultdict[int, Set[str]](
            set
        )
        # Synchronization for the RDT store.
        self._lock = threading.RLock()
        # Signal when an object becomes present in the object store.
        self._object_present_cv = threading.Condition(self._lock)
        # Signal when an object is freed from the object store.
        self._object_freed_cv = threading.Condition(self._lock)

    def has_object(self, obj_id: str) -> bool:
        with self._lock:
            existed = obj_id in self._rdt_store
            if existed:
                return len(self._rdt_store[obj_id]) > 0
            return existed

    def has_tensor(self, tensor: Any) -> bool:
        # Method only used for testing.
        with self._lock:
            return id(tensor) in self._tensor_to_object_ids

    def get_object(self, obj_id: str) -> Optional[List[Any]]:
        with self._lock:
            if self._rdt_store[obj_id][0].error:
                raise self._rdt_store[obj_id][0].error
            return self._rdt_store[obj_id][0].data

    def add_object(
        self,
        obj_id: str,
        rdt_object: Union[List[Any], Exception],
        is_primary: bool = False,
    ):
        """Add an RDT object to the RDT store.

        Args:
            obj_id: The object ID of the RDT object.
            rdt_object: A list of tensors representing the RDT object.
            is_primary: Whether the RDT object is the primary copy.
        """
        with self._object_present_cv:
            if isinstance(rdt_object, Exception):
                self._rdt_store[obj_id].append(
                    _RDTObject([], is_primary, error=rdt_object)
                )
            else:
                for tensor in rdt_object:
                    self._tensor_to_object_ids[id(tensor)].add(obj_id)
                # Append to the queue instead of overwriting
                self._rdt_store[obj_id].append(
                    _RDTObject(
                        rdt_object,
                        is_primary,
                    )
                )
            self._object_present_cv.notify_all()

    def add_object_primary(
        self, obj_id: str, tensors: List[Any], tensor_transport: str
    ) -> TensorTransportMetadata:
        with self._object_present_cv:
            # A primary entry may already exist from a prior attempt of the
            # same task (e.g., a task that succeeded and populated the RDT
            # store but whose reply was lost, then got retried). Keep the
            # existing primary — do not re-store — and return metadata
            # derived from it so the metadata matches what `__ray_send__`
            # will actually transmit.
            queue = self._rdt_store.get(obj_id)
            if queue:
                tensors_to_describe = queue[0].data
            else:
                self.add_object(obj_id, tensors, is_primary=True)
                tensors_to_describe = tensors

        tensor_transport_manager = get_tensor_transport_manager(tensor_transport)
        tensor_transport_meta = (
            tensor_transport_manager.extract_tensor_transport_metadata(
                obj_id, tensors_to_describe
            )
        )

        if tensor_transport_meta.tensor_meta and not device_match_transport(
            tensor_transport_meta.tensor_device, tensor_transport
        ):
            raise ValueError(
                f"Tensor transport backend {tensor_transport} does not support "
                f"tensor transfer on device {tensor_transport_meta.tensor_device}."
            )

        return tensor_transport_meta

    def is_primary_copy(self, obj_id: str) -> bool:
        with self._lock:
            return self.has_object(obj_id) and self._rdt_store[obj_id][0].is_primary

    def wait_and_get_object(
        self, obj_id: str, timeout: Optional[float] = None
    ) -> List[Any]:
        """Atomically waits for the RDT object to be present in the RDT
        store, then gets it. If the object is not present after the optional
        timeout, raise a TimeoutError.

        Args:
            obj_id: The object ID to wait for.
            timeout: The maximum time in seconds to wait for the object to be
                present in the RDT store. If not specified, wait indefinitely.

        Returns:
            The tensors in the RDT object.
        """
        with self._lock:
            self._wait_object(obj_id, timeout)
            return self.get_object(obj_id)

    def wait_and_pop_object(
        self, obj_id: str, timeout: Optional[float] = None
    ) -> List[Any]:
        """Atomically waits for the RDT object to be present in the RDT
        store, then pops it.  If the object is not present after the optional
        timeout, raise a TimeoutError.

        Args:
            obj_id: The object ID to wait for.
            timeout: The maximum time in seconds to wait for the object to be
                present in the RDT store. If not specified, wait indefinitely.

        Returns:
            The RDT object.
        """
        with self._lock:
            self._wait_object(obj_id, timeout)
            return self.pop_object(obj_id)

    def _wait_object(self, obj_id: str, timeout: Optional[float] = None) -> None:
        """Helper method to wait for the RDT object to be present in the RDT store.
        If the object is not present after the optional timeout, raise a
        TimeoutError.

        Args:
            obj_id: The object ID to wait for.
            timeout: The maximum time in seconds to wait for the object to be
                present in the RDT store. If not specified, wait indefinitely.
        """
        with self._object_present_cv:
            if not self._object_present_cv.wait_for(
                lambda: self.has_object(obj_id),
                timeout=timeout,
            ):
                raise TimeoutError(
                    f"ObjectRef({obj_id}) not found in RDT object store after {timeout}s, transfer may have failed. Please report this issue on GitHub: https://github.com/ray-project/ray/issues/new/choose"
                )

    def pop_object(self, obj_id: str) -> List[Any]:
        with self._lock:
            queue = self._rdt_store.get(obj_id)
            assert queue is not None, f"obj_id={obj_id} not found in RDT store"
            rdt_object = queue.popleft()
            if len(queue) == 0:
                del self._rdt_store[obj_id]
            if rdt_object.error:
                raise rdt_object.error
            for tensor in rdt_object.data:
                self._tensor_to_object_ids[id(tensor)].remove(obj_id)
                if len(self._tensor_to_object_ids[id(tensor)]) == 0:
                    self._tensor_to_object_ids.pop(id(tensor))
            self._object_freed_cv.notify_all()
            return rdt_object.data

    def wait_tensor_freed(self, tensor: Any, timeout: Optional[float] = None) -> None:
        """Wait for the object to be freed from the RDT store."""
        with self._object_freed_cv:
            if not self._object_freed_cv.wait_for(
                lambda: id(tensor) not in self._tensor_to_object_ids,
                timeout=timeout,
            ):
                raise TimeoutError(
                    f"Tensor {tensor} not freed from RDT object store after {timeout}s. The tensor will not be freed until all ObjectRefs containing the tensor have gone out of scope."
                )

    def get_num_objects(self) -> int:
        """Return the number of objects in the RDT store."""
        with self._lock:
            # Count total objects across all queues
            return sum(len(queue) for queue in self._rdt_store.values())

    def get_memory_info_snapshot(
        self, actor_id: str, node_id: str, pid: int
    ) -> RDTMemoryInfo:
        """Return a snapshot of this store's memory accounting.

        Runs a brief O(tensor_count) scan under the existing store lock that
        collects only immutable scalar metadata (device string, data pointer,
        byte count, numel, element size). Interval deduplication and sorting
        happen after releasing the lock. No tensor or storage references escape
        the method, so the query cannot extend tensor lifetime beyond this
        bounded call.

        Args:
            actor_id: The actor ID to record in the snapshot.
            node_id: The node ID to record in the snapshot.
            pid: The process ID to record in the snapshot.

        Returns:
            An :class:`RDTMemoryInfo` snapshot. An empty store returns accurate
            zero counts without importing torch.
        """
        # Check whether any non-error entry holds data to inspect. If not,
        # return an accurate zero snapshot without importing torch.
        with self._lock:
            has_data = any(
                entry.error is None and len(entry.data) > 0
                for queue in self._rdt_store.values()
                for entry in queue
            )
            if not has_data:
                return RDTMemoryInfo(
                    actor_id=actor_id,
                    node_id=node_id,
                    pid=pid,
                    devices=[],
                    num_error_entries=sum(
                        1
                        for queue in self._rdt_store.values()
                        for entry in queue
                        if entry.error is not None
                    ),
                    num_uninspectable_tensors=0,
                    incomplete=False,
                    unsupported=[],
                )

        torch = None
        try:
            import torch
        except ImportError:
            torch = None

        object_ids_per_device: Dict[str, Set[str]] = defaultdict(set)
        num_copies: Dict[str, int] = defaultdict(int)
        tensor_count: Dict[str, int] = defaultdict(int)
        tensor_bytes: Dict[str, int] = defaultdict(int)
        spans_per_device: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        num_error_entries = 0
        num_uninspectable_tensors = 0
        unsupported_categories: Set[str] = set()
        incomplete = False

        with self._lock:
            for obj_id, queue in self._rdt_store.items():
                for entry in queue:
                    if entry.error is not None:
                        num_error_entries += 1
                        continue
                    touched_devices: Set[str] = set()
                    for tensor in entry.data:
                        accounting, reason = _inspect_tensor(tensor, torch)
                        if accounting is None:
                            incomplete = True
                            num_uninspectable_tensors += 1
                            unsupported_categories.add(reason)
                            continue
                        device, start, end, tbytes = accounting
                        touched_devices.add(device)
                        tensor_count[device] += 1
                        tensor_bytes[device] += tbytes
                        spans_per_device[device].append((start, end))
                    for device in touched_devices:
                        object_ids_per_device[device].add(obj_id)
                        num_copies[device] += 1

        # Compute referenced-storage interval unions outside the lock.
        devices = []
        for device in sorted(spans_per_device.keys()):
            devices.append(
                RDTDeviceMemoryInfo(
                    device=device,
                    num_objects=len(object_ids_per_device[device]),
                    num_copies=num_copies[device],
                    tensor_count=tensor_count[device],
                    tensor_bytes=tensor_bytes[device],
                    referenced_storage_bytes=_union_span_bytes(
                        spans_per_device[device]
                    ),
                )
            )

        return RDTMemoryInfo(
            actor_id=actor_id,
            node_id=node_id,
            pid=pid,
            devices=devices,
            num_error_entries=num_error_entries,
            num_uninspectable_tensors=num_uninspectable_tensors,
            incomplete=incomplete,
            unsupported=sorted(unsupported_categories),
        )
