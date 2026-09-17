"""Small synthetic maps qualify serialization, never microscope physics."""
from io import BytesIO
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo
import stat

import numpy as np
import pytest

from temsim import input_io
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.physics.lens_field_provider import (
    load_magnetic_field_map, bind_imported_lens_field_map,
    resolve_runtime_lens_field_provider, lens_geometry_binding,
)
from temsim.working_point import WorkingPointCheckpoint, WorkingPointArchiveIndex
from temsim.working_point_export import make_portable_inputs


@pytest.mark.parametrize("suffix", (".npz", ".csv"))
def test_archived_map_retains_exact_fields_with_original_file_absent(tmp_path, suffix):
    state = default_state()
    lens = next(lens for lens in state.lenses if lens.key == "objective_lens")
    path = tmp_path / ("synthetic-map" + suffix)
    r = np.array([0., .001, .002])
    z = lens.z_mm * .001 + np.array([-.002, 0., .002])
    br = np.zeros((3, 3))
    bz = np.full((3, 3), .37)
    if suffix == ".npz":
        np.savez_compressed(path, r_m=r, z_m=z, br_t=br, bz_t=bz,
                            metadata_json='{"map_type":"axisymmetric_rz"}')
    else:
        rr, zz = np.meshgrid(r, z, indexing="ij")
        np.savetxt(path, np.column_stack((rr.ravel(), zz.ravel(), br.ravel(), bz.ravel())),
                   delimiter=",", header="r_m,z_m,br_t,bz_t", comments="")
    field_map = load_magnetic_field_map(path, geometry_binding=lens_geometry_binding(state, lens.key, lens),
                                       provenance_kind="fem", reference_excitation_percent=100., require_divergence=True)
    bind_imported_lens_field_map(state, lens.key, field_map, native_provider=lens)
    provider = resolve_runtime_lens_field_provider(state, lens.key, lens)
    expected = provider.magnetic_field_t(z * 1000.)
    snapshot = capture_instrument_snapshot(state)
    point = make_portable_inputs(WorkingPointCheckpoint(snapshot, {}, state.sample.z_mm, snapshot.physical_digest,
                                 {"package_kind": "INSTRUMENT_INPUTS_ONLY"}))
    package = tmp_path / "portable.temwp"
    point.write_package(package, mode="inputs")
    moved = tmp_path / ("original-unavailable" + suffix)
    assert path.resolve().is_relative_to(tmp_path.resolve()) and moved.resolve().is_relative_to(tmp_path.resolve())
    path.rename(moved)
    restored = WorkingPointArchiveIndex.read(package).load().compatible_state()
    restored_lens = next(lens for lens in restored.lenses if lens.key == "objective_lens")
    actual = resolve_runtime_lens_field_provider(restored, lens.key, restored_lens)
    assert actual.model_status == provider.model_status == "measured_or_fem_geometry_bound"
    assert actual.field_map.provenance == field_map.provenance
    assert actual.binding.geometry_fingerprint == provider.binding.geometry_fingerprint
    for a, b in zip(actual.field_map.axes_m + actual.field_map.components_t, field_map.axes_m + field_map.components_t, strict=True):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    assert np.array_equal(actual.magnetic_field_t(z * 1000.), expected)
    assert not path.exists()


def _archive_for_file(tmp_path, data, name="field.npz"):
    root = tmp_path / "configs"
    payload = dict(schema=input_io.ARCHIVE_SCHEMA, config_root=str(root), runtime=input_io.runtime_identity(),
                   files=[dict(path=str(root / name), config_relative=name, content_hex=data.hex(), sha256=sha256(data).hexdigest())])
    payload["digest"] = json_digest(payload)
    return payload


@pytest.mark.parametrize("kind", ("oversized", "truncated", "object", "traversal", "link", "duplicate"))
def test_inner_npz_is_checked_before_numpy_allocates(tmp_path, monkeypatch, kind):
    stream = BytesIO()
    dtype = "O" if kind == "object" else "<f8"
    shape = (10**12,) if kind == "oversized" else (4,)
    np.lib.format.write_array_header_1_0(stream, dict(descr=dtype, fortran_order=False, shape=shape))
    if kind != "truncated":
        stream.write(b"\0" * 32)
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        entry = ZipInfo("../outside.npy" if kind == "traversal" else "bz_t.npy")
        if kind == "link":
            entry.create_system = 3
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(entry, stream.getvalue())
        if kind == "duplicate":
            archive.writestr("bz_t.npy", stream.getvalue())
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Must reject before numeric allocation"))
    with pytest.raises(ValueError):
        input_io.InputArchive(_archive_for_file(tmp_path, output.getvalue()))
