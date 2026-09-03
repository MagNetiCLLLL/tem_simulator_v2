import math
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.optics.column import default_state
from temsim.specimen.sample_region import (
    SampleRegionElectronPath,
    _angular_acceptance,
    _electron_paths,
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


def test_boundary_input_path_ends_at_the_centred_specimen_entrance_face():
    state = default_state()
    sample_z_mm = float(state.sample.z_mm)
    entry_z_mm = sample_z_mm - 0.05
    slope_x = 0.01
    sample_x_m = 12.0e-9
    entry_x_m = sample_x_m + (entry_z_mm - sample_z_mm) * 1.0e-3 * slope_x
    simulation = SimpleNamespace(
        incident=SimpleNamespace(
            z=np.asarray((entry_z_mm, sample_z_mm)),
            alive=np.asarray((True,)),
            blocked_z=np.asarray((np.nan,)),
            x=np.asarray(((entry_x_m,), (sample_x_m,))),
            y=np.zeros((2, 1)),
            tx=np.full((2, 1), slope_x),
            ty=np.zeros((2, 1)),
            energy_offset_ev=np.zeros(1),
            ray_weight=np.ones(1),
        )
    )
    elastic = SimpleNamespace(trajectories=(), material_flights=())

    paths, _entry_bundle, _sample_bundle = _electron_paths(
        state,
        simulation,
        elastic,
        entry_z_mm=entry_z_mm,
        secondary_count=0,
        rng=np.random.default_rng(1),
    )

    endpoint = paths[0].positions_mm[-1]
    expected_top_z_mm = sample_z_mm - 0.5 * state.sample.thickness_nm * 1.0e-6
    assert endpoint[2] == pytest.approx(expected_top_z_mm)
    assert endpoint[0] == pytest.approx(
        -0.5 * state.sample.thickness_nm * slope_x * 1.0e-6
    )
