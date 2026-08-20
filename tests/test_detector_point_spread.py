from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog
from temsim.detector import plane_image
from temsim.detector.plane_image import detector_response_image
from temsim.detector.point_spread import (
    DetectorPointSpread,
    apply_point_spread,
)
from temsim.optics.column import default_state


def _point_spread(**updates):
    values = {
        "model": "gaussian",
        "sigma_x_mm": 2.0,
        "sigma_y_mm": 1.0,
        "rotation_deg": 0.0,
        "status": "provisional_model_parameter",
        "source": "unit_test",
    }
    values.update(updates)
    return DetectorPointSpread(**values)


def test_gaussian_point_spread_is_nonnegative_normalized_and_anisotropic():
    ideal = np.zeros((129, 129), dtype=float)
    ideal[64, 64] = 1.0

    response = apply_point_spread(
        ideal,
        _point_spread(),
        pixel_size_x_mm=0.25,
        pixel_size_y_mm=0.25,
    )

    yy, xx = np.indices(response.shape, dtype=float)
    xx = (xx - 64.0) * 0.25
    yy = (yy - 64.0) * 0.25
    total = float(response.sum())
    sigma_x = np.sqrt(float(np.sum(response * xx**2) / total))
    sigma_y = np.sqrt(float(np.sum(response * yy**2) / total))
    assert np.all(response >= 0.0)
    assert total == pytest.approx(1.0, abs=1.0e-12)
    assert sigma_x == pytest.approx(2.0, rel=2.0e-3)
    assert sigma_y == pytest.approx(1.0, rel=2.0e-3)


def test_none_point_spread_is_an_exact_identity():
    image = np.arange(25, dtype=float).reshape(5, 5)
    point_spread = _point_spread(
        model="none",
        sigma_x_mm=0.0,
        sigma_y_mm=0.0,
    )

    response = apply_point_spread(
        image,
        point_spread,
        pixel_size_x_mm=1.0,
        pixel_size_y_mm=1.0,
    )

    assert np.array_equal(response, image)
    assert response is not image


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"model": "lorentzian"}, "model must be one of"),
        ({"sigma_x_mm": 0.0}, "requires positive"),
        ({"sigma_y_mm": -1.0}, "cannot be negative"),
        ({"rotation_deg": np.nan}, "must be finite"),
        ({"status": "assumed_true"}, "status must be one of"),
        ({"source": ""}, "source must not be empty"),
    ),
)
def test_point_spread_rejects_invalid_parameters(updates, message):
    with pytest.raises(ValueError, match=message):
        _point_spread(**updates).validated()


def test_detector_response_applies_hit_mask_before_and_after_psf(monkeypatch):
    plane = SimpleNamespace(
        key="detector",
        name="Annular detector",
        z_mm=10.0,
        outer_width_mm=10.0,
        point_spread_model="gaussian",
        point_spread_sigma_x_mm=0.5,
        point_spread_sigma_y_mm=0.5,
        point_spread_rotation_deg=0.0,
        point_spread_status="provisional_model_parameter",
        point_spread_source="unit_test",
        hit_mask=lambda x, y: (
            (np.hypot(x, y) >= 1.0) & (np.hypot(x, y) <= 5.0)
        ),
    )
    monkeypatch.setattr(
        plane_image,
        "_column_plane_samples",
        lambda *_args: (
            np.array(((1.1, 0.0), (0.0, 0.0), (6.0, 0.0))),
            np.ones(3),
        ),
    )

    response = detector_response_image(
        object(),
        SimpleNamespace(recording_planes=[plane]),
        "detector",
        pixels=128,
    )

    assert response.accepted_weight == pytest.approx(1.0)
    assert 0.0 < response.response_weight < response.accepted_weight
    assert response.retained_fraction == pytest.approx(
        response.response_weight / response.accepted_weight
    )
    assert np.all(response.response_intensity >= 0.0)


def test_selected_manifest_owns_every_recording_plane_point_spread():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())

    for component in state.recording_planes:
        part = state._module_manifest_parts[component.key].data
        assembly_part = state._resolved_assembly.part(component.key)
        assert part["signal_collection_surface"] == (
            "upstream_top_surface"
        )
        assert part["optical_reference_local_z_mm"] == pytest.approx(
            part["local_start_z_mm"]
        )
        assert component.z_mm == pytest.approx(assembly_part.start_z_mm)
        assert component.point_spread_model == part["point_spread_model"]
        assert component.point_spread_sigma_x_mm == pytest.approx(
            part["point_spread_sigma_x_mm"]
        )
        assert component.point_spread_sigma_y_mm == pytest.approx(
            part["point_spread_sigma_y_mm"]
        )
        assert component.point_spread_rotation_deg == pytest.approx(
            part["point_spread_rotation_deg"]
        )
        assert component.point_spread_status == part["point_spread_status"]
        assert component.point_spread_source == part["point_spread_source"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "signal_collection_surface",
            "mechanical_centre",
            "upstream top surface",
        ),
        (
            "optical_reference_local_z_mm",
            1150.0,
            "signal plane must coincide with local_start_z_mm",
        ),
    ),
)
def test_manifest_rejects_non_top_detector_signal_plane(
    field, value, message
):
    document = deepcopy(module_manifest.read_document(
        module_manifest.MODULE_ROOT
        / "project_and_recording_system"
        / "NoEnergyFilter.toml"
    ))
    camera = next(
        part for part in document["parts"] if part["key"] == "camera"
    )
    camera[field] = value

    with pytest.raises(ValueError, match=message):
        module_manifest.validate_document(document)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("point_spread_model", "airy", "model must be one of"),
        ("point_spread_sigma_x_mm", 0.0, "requires positive sigma"),
        ("point_spread_sigma_y_mm", -1.0, "cannot be negative"),
        ("point_spread_rotation_deg", True, "must be finite numeric"),
        ("point_spread_status", "assumed_true", "status must be one of"),
        ("point_spread_source", "", "source must not be empty"),
    ),
)
def test_manifest_rejects_invalid_detector_point_spread(field, value, message):
    document = deepcopy(module_manifest.read_document(
        module_manifest.MODULE_ROOT
        / "project_and_recording_system"
        / "NoEnergyFilter.toml"
    ))
    camera = next(
        part for part in document["parts"] if part["key"] == "camera"
    )
    camera[field] = value

    with pytest.raises(ValueError, match=message):
        module_manifest.validate_document(document)
