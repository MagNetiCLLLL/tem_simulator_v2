"""Value-keyed ownership of static Ray Diagram graphics (no physics cache)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class _Layer:
    signature: tuple
    items: tuple
    records: dict[str, tuple]


class StaticRayLayers:
    """Reuse graphics and their diagnostic records; release removed layers.

    Signatures contain only detached drawing inputs, never mutable result
    objects. Builders append to the workspace's established inspection lists.
    These lists are reassembled each publication without recreating Qt items.
    """

    RECORD_NAMES = (
        "component_marker_items", "_component_labels", "_ray_label_items",
        "sample_marker_items", "aperture_marker_items",
        "aperture_optical_plane_items", "aperture_stop_segment_items",
        "recording_surface_range_items", "_aperture_span_records",
        "_aperture_projection_records", "deflector_pair_items",
        "crossover_marker_items", "column_wall_items",
    )

    def __init__(self) -> None:
        self.layers: dict[tuple, _Layer] = {}
        self.rebuilt = 0
        self.reused = 0
        self._wanted: set[tuple] = set()

    def begin(self, workspace) -> None:
        self._wanted = set()
        for name in self.RECORD_NAMES:
            setattr(workspace, name, [])

    def update(
        self, workspace, key: tuple, signature: tuple, build: Callable[[], None]
    ) -> None:
        self._wanted.add(key)
        old = self.layers.get(key)
        if old is not None and old.signature == signature:
            for name, values in old.records.items():
                getattr(workspace, name).extend(values)
            self.reused += 1
            return
        before_items = set(workspace.plot.plotItem.items)
        lengths = {name: len(getattr(workspace, name)) for name in self.RECORD_NAMES}
        try:
            build()
        except Exception:
            # Keep the previous complete layer if a replacement cannot draw.
            for item in tuple(workspace.plot.plotItem.items):
                if item not in before_items:
                    workspace.plot.removeItem(item)
            for name, length in lengths.items():
                del getattr(workspace, name)[length:]
            raise
        self.remove(workspace, key)
        self.layers[key] = _Layer(
            signature,
            tuple(item for item in workspace.plot.plotItem.items if item not in before_items),
            {name: tuple(getattr(workspace, name)[length:]) for name, length in lengths.items()},
        )
        self.rebuilt += 1

    def remove(self, workspace, key: tuple) -> None:
        layer = self.layers.pop(key, None)
        if layer is not None:
            for item in layer.items:
                workspace.plot.removeItem(item)

    def finish(self, workspace) -> None:
        for key in tuple(self.layers):
            if key not in self._wanted:
                self.remove(workspace, key)
