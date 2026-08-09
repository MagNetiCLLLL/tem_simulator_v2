import tomllib
from pathlib import Path

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state
from temsim.optics.magnetic_lens_aberration import spherical_aberration_mm
from temsim.physics.core import complex_transfer, propagate


ROOT = Path(__file__).resolve().parents[1]
LATEST_IMAGE_SEQUENCE = (
    "image_ol_post_lens",
    "image_hpol_hexapole",
    "image_qpol_quadrupole",
    "image_dp11_deflector",
    "image_tl11_lens",
    "image_dp12_deflector",
    "image_tl12_lens",
    "image_dph1_deflector",
    "image_hp1_hexapole",
    "image_dp21_deflector",
    "image_tl21_lens",
    "image_dp22_deflector",
    "image_tl22_lens",
    "image_dph2_deflector",
    "image_hp2_hexapole",
    "image_adapter_lens",
    "image_ish_deflector",
    "image_dsh_deflector",
    "image_dstg_quadrupole",
    "image_sad_plane",
)


def _combined_state():
    state = default_state()
    AssemblyCatalog().apply(
        state,
        AssemblySelection(
            "FEG",
            "C3 + Probe Corrector + Image Corrector",
            "Energy Filter",
        ),
    )
    state.step_mm = 0.1
    state.history_step_mm = 0.1
    return state


@pytest.mark.parametrize(
    "filename",
    ("C3_ImageCorrector.toml", "C3_ProbeCorrector_ImageCorrector.toml"),
)
def test_latest_image_corrector_channels_keep_existing_centres(filename):
    path = ROOT / "configs" / "instruments" / "column" / filename
    parts = tomllib.loads(path.read_text(encoding="utf-8"))["parts"]
    image_parts = [
        part for part in sorted(parts, key=lambda item: item["order"])
        if str(part["key"]).startswith("image_")
        and not bool(part.get("mechanical_only", False))
    ]
    keys = tuple(part["key"] for part in image_parts)
    start = keys.index(LATEST_IMAGE_SEQUENCE[0])
    assert keys[start:start + len(LATEST_IMAGE_SEQUENCE)] == (
        LATEST_IMAGE_SEQUENCE
    )
    by_key = {part["key"]: part for part in parts}
    assert by_key["image_tl12_lens"]["local_center_z_mm"] == (
        by_key["image_dp12_deflector"]["local_center_z_mm"]
    )
    assert by_key["image_dph1_deflector"]["local_center_z_mm"] == (
        by_key["image_hp1_hexapole"]["local_center_z_mm"]
    )
    assert by_key["image_dph2_deflector"]["local_center_z_mm"] == (
        by_key["image_hp2_hexapole"]["local_center_z_mm"]
    )


def test_both_main_hexapole_relays_are_unit_conjugates():
    state = _combined_state()
    by_key = {
        component.key: component
        for component in (*state.lenses, *state.corrector_elements)
    }
    for upstream, downstream in (
        ("probe_hp2_hexapole", "probe_hp1_hexapole"),
        ("image_hp1_hexapole", "image_hp2_hexapole"),
    ):
        matrix = complex_transfer(
            state, by_key[upstream].z_mm, by_key[downstream].z_mm
        )
        assert abs(matrix[0, 1]) < 2.0e-4
        assert abs(abs(matrix[0, 0]) - 1.0) < 5.0e-3

    relay = complex_transfer(
        state,
        state.objective_image_plane_z_mm,
        by_key["image_sad_plane"].z_mm,
    )
    assert abs(relay[0, 1]) < 1.0e-3
    assert abs(abs(relay[0, 0]) - 1.0) < 2.0e-2


def _trace_aberration(state, z0, z1, x, tx, y, ty, *, cs, hexapole):
    return propagate(
        state, z0, z1, x, tx, y, ty,
        include_spherical_aberration=cs,
        include_hexapole=hexapole,
    )


def _errors(result, baseline):
    return (
        result[1][-1] - baseline[1][-1]
        + 1j * (result[3][-1] - baseline[3][-1])
    )


def _assert_negative_corrector_contribution(
    state, z0, z1, x, tx, y, ty, main_hexapoles
):
    baseline = _trace_aberration(
        state, z0, z1, x, tx, y, ty, cs=False, hexapole=False
    )
    positive = _trace_aberration(
        state, z0, z1, x, tx, y, ty, cs=True, hexapole=False
    )
    saved_cs = [getattr(lens, "cs_mm", None) for lens in state.lenses]
    try:
        for lens in state.lenses:
            lens.cs_mm = 0.0
        correction = _trace_aberration(
            state, z0, z1, x, tx, y, ty, cs=True, hexapole=True
        )
    finally:
        for lens, value in zip(state.lenses, saved_cs):
            lens.cs_mm = value
    net = _trace_aberration(
        state, z0, z1, x, tx, y, ty, cs=True, hexapole=True
    )
    positive_error = _errors(positive, baseline)
    correction_error = _errors(correction, baseline)
    net_error = _errors(net, baseline)
    assert all(hexapole.strength_m3 > 0.0 for hexapole in main_hexapoles)
    assert np.vdot(positive_error, correction_error).real < 0.0
    assert np.sqrt(np.mean(abs(net_error) ** 2)) < (
        0.20 * np.sqrt(np.mean(abs(positive_error) ** 2))
    )


def test_all_round_lenses_are_positive_and_both_correctors_cancel_cs():
    state = _combined_state()
    by_key = {
        component.key: component
        for component in (*state.lenses, *state.corrector_elements)
    }
    assert all(
        spherical_aberration_mm(lens, state.beam_voltage_kv) > 0.0
        for lens in state.lenses if lens.enabled
    )

    phi = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
    alpha = 1.0e-3
    zero = np.zeros(phi.size)
    _assert_negative_corrector_contribution(
        state,
        state.sample.z_mm,
        by_key["image_sad_plane"].z_mm,
        zero,
        alpha * np.cos(phi),
        zero,
        alpha * np.sin(phi),
        (by_key["image_hp1_hexapole"], by_key["image_hp2_hexapole"]),
    )

    radius = 1.0e-4
    probe_hp2 = by_key["probe_hp2_hexapole"]
    probe_start = (
        probe_hp2.z_mm
        - 5.0 * probe_hp2.effective_length_mm / 2.355
    )
    _assert_negative_corrector_contribution(
        state,
        probe_start,
        state.sample.z_mm,
        radius * np.cos(phi),
        zero,
        radius * np.sin(phi),
        zero,
        (probe_hp2, by_key["probe_hp1_hexapole"]),
    )
