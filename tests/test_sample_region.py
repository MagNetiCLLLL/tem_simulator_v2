import math

import numpy as np
import pytest

from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.specimen.sample_region import (
    SampleRegionElectronPath,
    _angular_acceptance,
    _isotropic_directions,
)


@pytest.fixture
def detector_geometry():
    return EDSDetectorArrayGeometry(
        system_key="eds",
        segment_count=6,
        azimuth_centers_deg=(0.0, 60.0, 120.0, 180.0, 240.0, 300.0),
        takeoff_angle_deg=32.06,
        minimum_unshadowed_solid_angle_sr=4.45,
        analytical_holder_solid_angle_sr=4.04,
        windowless=True,
    )


def test_isotropic_photon_sampler_is_uniform_in_solid_angle():
    directions = _isotropic_directions(np.random.default_rng(71), 200_000)

    assert np.linalg.norm(directions, axis=1) == pytest.approx(
        np.ones(len(directions)), abs=2.0e-15
    )
    assert float(np.mean(directions[:, 2])) == pytest.approx(0.0, abs=0.004)
    assert float(np.mean(directions[:, 2] ** 2)) == pytest.approx(
        1.0 / 3.0, abs=0.004
    )


def test_angular_acceptance_matches_configured_aggregate_solid_angle(
    detector_geometry,
):
    directions = _isotropic_directions(np.random.default_rng(13), 250_000)
    accepted = np.asarray(
        [
            _angular_acceptance(
                direction,
                detector_geometry,
                detector_geometry.analytical_holder_solid_angle_sr,
            )[0]
            for direction in directions
        ],
        dtype=bool,
    )

    expected = detector_geometry.analytical_holder_solid_angle_sr / (
        4.0 * math.pi
    )
    assert float(np.mean(accepted)) == pytest.approx(expected, abs=0.0025)


def test_secondary_marker_can_be_explicitly_zero_weight_and_local_only():
    marker = SampleRegionElectronPath(
        positions_mm=np.asarray(((0.0, 0.0, 1.0), (0.0, 1.0e-4, 1.0))),
        kind="secondary_candidate",
        weight=0.0,
        kinetic_energy_ev=None,
        provenance="qualitative test marker",
        downstream_eligible=False,
    )

    assert marker.weight == 0.0
    assert marker.kinetic_energy_ev is None
    assert not marker.downstream_eligible
