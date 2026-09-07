"""Small helpers for immutable, deterministic JSON-like records.

Calculation manifests, design recipes and persistent artifact metadata share
the same deliberately narrow value model.  Keeping it here avoids subtly
different hashes for the same scientific inputs and rejects objects such as
NumPy arrays that would otherwise retain large mutable calculation results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType


def freeze_json(value: object) -> object:
    """Return a recursively immutable JSON-compatible representation."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        converted = float(value)
        if not (-float("inf") < converted < float("inf")):
            raise ValueError("Immutable JSON values must be finite")
        return converted
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return MappingProxyType({
            field.name: freeze_json(getattr(value, field.name))
            for field in fields(value)
        })
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for raw_key, item in sorted(
            value.items(), key=lambda pair: str(pair[0])
        ):
            key = str(raw_key)
            if key in frozen:
                raise ValueError(
                    f"Mapping keys collide after normalization: {key!r}"
                )
            frozen[key] = freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, (set, frozenset)):
        frozen = tuple(freeze_json(item) for item in value)
        return tuple(sorted(frozen, key=repr))

    value_type = type(value)
    if value_type.__module__.split(".", 1)[0] == "numpy":
        ndim = getattr(value, "ndim", None)
        if ndim == 0 and hasattr(value, "item"):
            return freeze_json(value.item())
        raise TypeError("Immutable JSON records cannot contain NumPy arrays")
    raise TypeError(
        "Immutable JSON records support only JSON-like values, not "
        f"{value_type.__name__}"
    )


def thaw_json(value: object) -> object:
    """Return a detached JSON-serializable copy of a frozen value."""

    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Encode one value using the project's canonical hashing format."""

    return json.dumps(
        thaw_json(freeze_json(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def json_digest(value: object) -> str:
    """Return the SHA-256 digest of a canonical JSON-like value."""

    return sha256(canonical_json_bytes(value)).hexdigest()


__all__ = (
    "canonical_json_bytes",
    "freeze_json",
    "json_digest",
    "thaw_json",
)
