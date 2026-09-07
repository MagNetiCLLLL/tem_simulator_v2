"""Retained-memory estimates for shared numeric calculation products.

These are ownership estimates, not process RSS or device-memory measurements.
Mapped arrays are charged at their logical storage size, conservatively.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import mmap
import sys

import numpy as np


def retained_memory_inventory(*roots: object) -> dict[int, int]:
    """Count Python owners and shared NumPy buffers once, including views."""

    inventory: dict[int, int] = {}

    def visit(value: object) -> None:
        if value is None or id(value) in inventory:
            return
        inventory[id(value)] = int(sys.getsizeof(value))
        if isinstance(value, np.ndarray):
            if value.base is not None:
                visit(value.base)
            elif not value.flags.owndata:
                # A mapped array has an external OS buffer whose Python
                # wrapper does not report its logical byte span.
                inventory[id(value)] += int(value.nbytes)
            if value.dtype.hasobject:
                for item in value.flat:
                    visit(item)
        elif isinstance(value, mmap.mmap):
            inventory[id(value)] += len(value)
        elif isinstance(value, memoryview):
            visit(value.obj)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item)
        elif is_dataclass(value) and not isinstance(value, type):
            attributes = getattr(value, "__dict__", None)
            if isinstance(attributes, dict):
                visit(attributes)
            else:
                for field in fields(value):
                    visit(getattr(value, field.name))
        else:
            attributes = getattr(value, "__dict__", None)
            if isinstance(attributes, dict):
                visit(attributes)

    for root in roots:
        visit(root)
    return inventory


def estimate_result_cache_bytes(*results: object) -> int:
    """Estimate all retained roots together without double-counting buffers."""

    return sum(retained_memory_inventory(*results).values())


class RetainedMemoryLedger:
    """Incremental metadata-only accounting for immutable cached products.

    Owners must replace an entry after enriching a result. Inventory entries
    hold only object IDs and sizes, never additional references to results.
    """

    def __init__(self) -> None:
        self._entries: dict[object, dict[int, int]] = {}
        self._owners: dict[int, tuple[int, int]] = {}
        self.total_bytes = 0

    def replace(self, key: object, value: object) -> int:
        inventory = retained_memory_inventory(value)
        return self.replace_inventory(key, inventory)

    def replace_inventory(self, key: object, inventory: dict[int, int]) -> int:
        """Install an inventory already checked against an admission budget."""

        self.remove(key)
        self._entries[key] = inventory
        for object_id, size in inventory.items():
            previous = self._owners.get(object_id)
            if previous is None:
                self._owners[object_id] = (1, size)
                self.total_bytes += size
            else:
                count, previous_size = previous
                # Shared metadata may have been enriched. Use the larger
                # estimate until its last cache owner is removed.
                retained_size = max(previous_size, size)
                self._owners[object_id] = (count + 1, retained_size)
                self.total_bytes += retained_size - previous_size
        return sum(inventory.values())

    def remove(self, key: object) -> None:
        for object_id in self._entries.pop(key, {}):
            count, size = self._owners[object_id]
            if count == 1:
                self.total_bytes -= size
                del self._owners[object_id]
            else:
                self._owners[object_id] = (count - 1, size)

    def clear(self) -> None:
        self._entries.clear()
        self._owners.clear()
        self.total_bytes = 0
