"""Photon geometry reuse never reuses energy-dependent attenuation or counts."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector import eds_photon_transport as transport
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry


class _Material:
    key = "energy-dependent-test-material"
    density_g_cm3 = 2.0

    def __init__(self):
        self.energies = []

    def mass_attenuation_cm2_g(self, energy):
        self.energies.append(float(energy))
        return float("inf") if energy < 500 else 1000.0 / energy


class _CountedOccluder:
    def __init__(self, base):
        self.base = base
        self.key = base.key
        self.calls = 0

    def ray_intervals(self, *args):
        self.calls += 1
        return self.base.ray_intervals(*args)


def _geometry():
    return EDSDetectorArrayGeometry(
        system_key="eds", segment_count=2, azimuth_centers_deg=(0., 180.),
        takeoff_angle_deg=32., minimum_unshadowed_solid_angle_sr=1.,
        analytical_holder_solid_angle_sr=1., windowless=True,
    )


def _surfaces():
    return tuple(transport.PlanarEDSDetectorSegment(
        segment_index=index, center_mm=(sign * 10., 0., 0.),
        normal=(-sign, 0., 0.), u_axis=(0., 1., 0.),
        active_shape="circle", active_size_mm=(4., 4.),
        efficiency=efficiency, provenance="synthetic test geometry",
    ) for index, sign, efficiency in ((0, 1., 0.6), (1, -1., 0.8)))


def _sample(material):
    return _CountedOccluder(transport.FiniteSpecimenOccluder(
        key="sample", centre_global_mm=(0., 0., 0.), size_local_mm=(1., 1., 1.),
        rotation_local_to_global=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
        envelope_shape="rectangle", material=material,
    ))


def _rays(*, sourced):
    takeoff = np.deg2rad(32.)
    directions = ((1., 0., 0.), (-1., 0., 0.)) if sourced else (
        (np.cos(takeoff), 0., -np.sin(takeoff)),
        (-np.cos(takeoff), 0., -np.sin(takeoff)),
    )
    rays = []
    for geometry_index, direction in enumerate((*directions, (0., 0., 1.))):
        for transition, energy, weight in (("low", 400., 2.), ("K", 1740., 3.), ("high", 8000., 0.)):
            rays.append(transport.EDSPhotonRay(
                photon_id=f"g{geometry_index}:{transition}",
                origin_mm=(0., 0., 0.), direction=direction,
                energy_ev=energy, statistical_weight=weight,
                source_key="sample", transition=transition,
                emission_key=f"emission:{transition}",
            ))
    return tuple(rays)


@pytest.mark.parametrize("sourced", [False, True])
@pytest.mark.parametrize("stored_limit", [None, 1])
def test_cached_population_equals_uncached_paths_metrics_order_and_counts(monkeypatch, sourced, stored_limit):
    material = _Material()
    sample = _sample(material)
    rays = _rays(sourced=sourced)
    kwargs = dict(
        detector_surfaces=_surfaces() if sourced else (),
        holder_occluders=(sample,), include_specimen=False, include_support=False,
        aggregate_detector_efficiency=0.7, maximum_stored_paths=stored_limit,
    )
    cached = transport.transport_eds_photons(SimpleNamespace(), iter(rays), _geometry(), **kwargs)
    assert sample.calls == 3
    assert set(material.energies) == {400., 1740., 8000.}
    # Even outside/zero-weight photons retain their own transport semantics.
    assert cached.metrics["photon_count"] == len(rays)
    assert cached.metrics["blocked_photon_count"] == 3
    monkeypatch.setattr(transport, "_PHOTON_GEOMETRY_CACHE_ENTRIES", 0)
    sample.calls = 0
    uncached = transport.transport_eds_photons(SimpleNamespace(), iter(rays), _geometry(), **kwargs)
    assert sample.calls == len(rays)
    assert cached == uncached
    direct = tuple(transport.trace_eds_photon(
        ray, _geometry(), detector_surfaces=kwargs["detector_surfaces"],
        occluders=(sample,), aggregate_detector_efficiency=0.7,
    ) for ray in rays)
    expected_paths = direct[:stored_limit] if stored_limit is not None else direct
    assert cached.paths == expected_paths


def test_exact_geometry_keys_are_not_rounded_and_lru_is_bounded(monkeypatch):
    monkeypatch.setattr(transport, "_PHOTON_GEOMETRY_CACHE_ENTRIES", 2)
    sample = _sample(_Material())
    # A,B,A,C,B: promoting A evicts B. Adjacent floating-point origins differ.
    origin_a = 0.2
    origin_b = np.nextafter(origin_a, 1.0)
    rays = tuple(transport.EDSPhotonRay(
        photon_id=str(index), origin_mm=(origin, 0., 0.), direction=(1., 0., 0.),
        energy_ev=1740., statistical_weight=1.,
    ) for index, origin in enumerate((origin_a, origin_b, origin_a, 0.3, origin_b)))
    result = transport.transport_eds_photons(
        SimpleNamespace(), rays, _geometry(), detector_surfaces=_surfaces(),
        holder_occluders=(sample,), include_specimen=False, include_support=False,
    )
    assert sample.calls == 4
    assert tuple(path.photon for path in result.paths) == rays


def test_cache_never_crosses_acquisitions_or_material_changes():
    rays = _rays(sourced=True)
    material_one = _Material()
    material_two = _Material()
    material_two.density_g_cm3 = 5.
    results = []
    for material in (material_one, material_two):
        sample = _sample(material)
        results.append(transport.transport_eds_photons(
            SimpleNamespace(), rays, _geometry(), detector_surfaces=_surfaces(),
            holder_occluders=(sample,), include_specimen=False, include_support=False,
        ))
        assert sample.calls == 3
    assert results[0].metrics["total_detected_weight"] > results[1].metrics["total_detected_weight"]


def test_support_geometry_is_resolved_once_per_exact_path(monkeypatch):
    scene = SimpleNamespace()
    monkeypatch.setattr(transport.SpecimenScene, "from_state", lambda *_args, **_kwargs: scene)
    calls = []
    holder = transport.AnnularPlanarLayerOccluder(
        key="support:bar", z_start_mm=0.1, z_end_mm=0.2,
        outer_radius_mm=1., hard_shadow=True,
    )

    def support(_state, ray, **_kwargs):
        calls.append((ray.origin_mm, ray.direction))
        return holder

    monkeypatch.setattr(transport, "support_occluder_for_ray", support)
    rays = tuple(transport.EDSPhotonRay(
        photon_id=str(index), origin_mm=(0., 0., 0.), direction=(0., 0., 1.),
        energy_ev=energy, statistical_weight=float(index + 1),
    ) for index, energy in enumerate((1740., 8000., 15000.)))
    cached = transport.transport_eds_photons(
        SimpleNamespace(), rays, _geometry(), include_specimen=False, include_support=True,
    )
    assert len(calls) == 1
    assert cached.metrics["blocked_by_component_counts"] == {"support:bar": 3}
    assert all(path.terminal_status == "blocked_by:support:bar" for path in cached.paths)
    monkeypatch.setattr(transport, "_PHOTON_GEOMETRY_CACHE_ENTRIES", 0)
    uncached = transport.transport_eds_photons(
        SimpleNamespace(), rays, _geometry(), include_specimen=False, include_support=True,
    )
    assert len(calls) == 4
    assert cached == uncached


def test_empty_photon_population_remains_empty():
    result = transport.transport_eds_photons(
        SimpleNamespace(), (), _geometry(), include_specimen=False, include_support=False,
    )
    assert result.paths == ()
    assert result.metrics["photon_count"] == 0
    assert result.metrics["total_detected_weight"] == 0.
