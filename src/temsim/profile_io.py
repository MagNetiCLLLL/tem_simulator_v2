"""TOML persistence for operating parameters."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import tomllib

import tomli_w

from temsim.assembly_catalog import AssemblySelection
from temsim.runtime_parameters import (
    editable_parameters,
    runtime_targets,
    validate_runtime_assignment,
)


def _atomic_write_profile(path: Path, document: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            tomli_w.dump(document, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_profile(path: str | Path, state, selection: AssemblySelection) -> None:
    devices = {}
    for key, target in runtime_targets(state).items():
        values = {
            parameter.name: parameter.value
            for parameter in editable_parameters(target)
            if parameter.value is not None
        }
        if values:
            devices[key] = values
    document = {
        "format_version": 1,
        "assembly": {
            "gun": selection.gun,
            "column": selection.column,
            "recording": selection.recording,
        },
        "devices": devices,
    }
    _atomic_write_profile(Path(path), document)


def read_profile(path: str | Path) -> tuple[AssemblySelection, dict]:
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    if not isinstance(document, dict):
        raise ValueError("Operating profile must be a TOML table")
    if int(document.get("format_version", 0)) != 1:
        raise ValueError("Unsupported operating-profile format")
    assembly = document.get("assembly")
    if not isinstance(assembly, dict):
        raise ValueError("Operating profile is missing the assembly table")
    selection = AssemblySelection(
        gun=str(assembly["gun"]),
        column=str(assembly["column"]),
        recording=str(assembly["recording"]),
    )
    devices = document.get("devices", {})
    if not isinstance(devices, dict):
        raise ValueError("Operating profile devices must be a table")
    return selection, dict(devices)


def apply_profile_values(state, values: dict) -> list[str]:
    if not isinstance(values, dict):
        raise ValueError("Operating profile devices must be a table")
    targets = runtime_targets(state)
    skipped = []
    pending = []
    for key, attributes in values.items():
        target = targets.get(key)
        if target is None:
            skipped.append(key)
            continue
        if not isinstance(attributes, dict):
            raise ValueError(f"Operating profile device {key} must be a table")
        allowed = {parameter.name for parameter in editable_parameters(target)}
        for name, value in attributes.items():
            if name not in allowed:
                skipped.append(f"{key}.{name}")
                continue
            converted = validate_runtime_assignment(target, name, value)
            pending.append((target.obj, name, converted))
    for obj, name, value in pending:
        setattr(obj, name, value)
    return skipped
