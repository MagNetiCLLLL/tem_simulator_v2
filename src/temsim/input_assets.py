"""Content-addressed immutable input buffers with explicit request ownership."""
from collections import OrderedDict
from hashlib import sha256
import math
from threading import RLock
import weakref

import numpy as np


def _bytes_owner(array):
    owner = array
    while isinstance(owner, np.ndarray):
        owner = owner.base
    return owner if isinstance(owner, bytes) else None


def freeze_numeric_array(values):
    array = np.asarray(values)
    if array.dtype.kind not in "biufc" or array.dtype.fields is not None:
        raise ValueError("Input assets require plain numeric arrays")
    owner = _bytes_owner(array)
    data = owner if owner is not None and array.flags.c_contiguous and array.nbytes == len(owner) else array.tobytes(order="C")
    return np.frombuffer(data, dtype=array.dtype).reshape(array.shape)


class InputAssetStore:
    """LRU retention; active leases pin unique immutable bytes exactly once."""
    def __init__(self, *, budget_bytes=256*1024**2):
        self._lock = RLock()
        self._entries = OrderedDict()
        self._pins = {}
        self._memo = {}
        self._resident = 0
        self._hashed_bytes = 0
        self.configure(budget_bytes)

    def configure(self, budget_bytes):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("Input asset retention budget must be a nonnegative integer")
        with self._lock:
            self.budget_bytes = budget_bytes
            self._evict()

    def capture(self):
        return InputAssetLease(self)

    def _evict(self):
        for key in tuple(self._entries):
            if self._resident <= self.budget_bytes:
                break
            if self._pins.get(key, 0) == 0:
                self._resident -= len(self._entries.pop(key))
                self._pins.pop(key, None)

    def _release(self, keys):
        with self._lock:
            for key in keys:
                self._pins[key] -= 1
            self._evict()

    def _admit(self, value, lease):
        value = np.asarray(value)
        if value.dtype.kind not in "biufc" or value.dtype.fields is not None:
            raise ValueError("Input assets require plain numeric arrays")
        descriptor = (value.dtype.str, value.shape, value.strides)
        immutable = not value.flags.writeable and _bytes_owner(value) is not None
        with self._lock:
            memo = self._memo.get(id(value)) if immutable else None
            if memo is not None and memo[0]() is value and memo[1] == descriptor and memo[2] in self._entries:
                key = memo[2]
            else:
                owner = _bytes_owner(value)
                data = owner if owner is not None and value.flags.c_contiguous and value.nbytes == len(owner) else value.tobytes(order="C")
                key = sha256(data).hexdigest()
                self._hashed_bytes += len(data)
                if key not in self._entries:
                    self._entries[key] = data
                    self._resident += len(data)
                if immutable:
                    value_id = id(value)
                    def forget(ref):
                        with self._lock:
                            if self._memo.get(value_id, (None,))[0] is ref:
                                self._memo.pop(value_id, None)
                    self._memo[value_id] = (weakref.ref(value, forget), descriptor, key)
            self._entries.move_to_end(key)
            if key not in lease._items:
                lease._items[key] = self._entries[key]
                self._pins[key] = self._pins.get(key, 0) + 1
            self._evict()
            return key

    def statistics(self):
        with self._lock:
            pinned = sum(len(data) for key, data in self._entries.items() if self._pins.get(key, 0))
            return dict(entries=len(self._entries), resident_bytes=self._resident, pinned_bytes=pinned,
                budget_bytes=self.budget_bytes, pinned_over_budget_bytes=max(0, pinned-self.budget_bytes),
                hashed_bytes=self._hashed_bytes)

    def retained_buffers(self):
        """Stable buffer owners for shared resource accounting; no copy."""
        with self._lock:
            return tuple(self._entries.values())


class InputAssetLease:
    """Pin while capturing/preparing; decoded arrays own their backing bytes."""
    def __init__(self, store):
        self._store = store
        self._items = {}
        self._closed = False
        self._finalizer = weakref.finalize(self, store._release, self._items)

    def register(self, value):
        if self._closed:
            raise ValueError("Input asset lease is closed")
        return self._store._admit(value, self)

    def array(self, key, dtype, shape, *, readonly):
        if self._closed or key not in self._items:
            raise ValueError("Missing pinned input asset")
        dtype = np.dtype(dtype)
        if (dtype.kind not in "biufc" or dtype.fields is not None or len(shape) > 32
                or any(type(n) is not int or n < 0 for n in shape)
                or math.prod(shape)*dtype.itemsize != len(self._items[key])):
            raise ValueError("Input asset shape/dtype does not match its immutable content")
        result = np.frombuffer(self._items[key], dtype=dtype).reshape(shape)
        return result if readonly else result.copy()

    def close(self):
        if not self._closed:
            self._closed = True
            self._finalizer()
            self._items.clear()


INPUT_ASSETS = InputAssetStore()
