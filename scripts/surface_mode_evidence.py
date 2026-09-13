"""Output-only complete-mode evidence, never an active source import API."""
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from temsim.immutable_json import thaw_json


def preserve_mode(directory, mode, near, record, radial, history, identity):
    directory = Path(directory)
    path = directory/f"completed-mode-{identity['mode_index']}.npz"
    if path.exists():
        raise FileExistsError(path)
    metadata = {"scope": "COMPLETE_SINGLE_ENERGY_NOT_COMPLETE_SOURCE_OR_IMAGE_ACCEPTANCE",
        "identity": identity, "record": record, "radial_representation": radial,
        "mode": {"mode_id": mode.mode_id, "weight": mode.weight_per_reference_electron,
                 "energy_kev": mode.energy_kev, "reference_plane": mode.reference_plane,
                 "axial_reference": asdict(mode.axial_reference), "scattering_history": mode.scattering_history},
        "near": {"energy_ev": near.energy_ev, "weight": near.weight,
                 "flux": near.flux, "phase_reference": near.phase_reference},
        "history_scope": "Physical complex boundary coefficients include the axial carrier. Covariant derivatives are projected physical normal derivatives. Duplicate Z planes retain both sides of a physical stop."}
    arrays = {"near_amplitude": near.amplitude, **{"history_"+key: value for key, value in history.items()}}
    for key in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        value = getattr(mode.plane, key)
        if value is not None:
            arrays["exit_"+key] = value
    arrays["metadata_json"] = np.asarray(json.dumps(thaw_json(metadata), allow_nan=False))
    # A distinct partial file remains unindexed after a failed write. An
    # atomic hard link refuses an existing destination on every platform.
    with tempfile.NamedTemporaryFile(dir=directory, prefix="pending-mode-", suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            digest.update(block)
    return {"file": path.name, "sha256": digest.hexdigest(), "mode_index": identity["mode_index"],
            "dependency_digest": identity["dependency_digest"], "scope": metadata["scope"]}
