"""Pure archive fixtures: preserve EDS records without running particle physics."""
import json
from zipfile import ZipFile

import numpy as np
import pytest

from temsim.detector.eds_signal import (
    EDSLineSignal, EDSMaterial, EDSMaterialQuadrature, EDSSpectrum,
    EDSVacancySignal, ElectronTrackSegment,
)
from temsim.detector.eds_photon_transport import (
    EDSPhotonPathResult, EDSPhotonRay, EDSPhotonTransportResult, PhotonMaterialInterval,
)
from temsim.immutable_json import thaw_json
from temsim.particle_section_io import _data_classes, _pack, _same_executed_record, _unpack
from temsim.physics.particle_sections import MaterialSectionCache
from temsim.specimen.elastic_transport import (
    ElasticMaterialFlight, ElasticScatterEvent, ElasticTerminalBundle,
    ElasticTrajectory, ElasticTransportResult,
)
from temsim.working_point import WorkingPointCheckpoint


@pytest.fixture
def material_cache():
    material = EDSMaterial("fixture-si", "Silicon fixture", 2.33, ((14, 1.0),), "archive fixture")
    track = ElectronTrackSegment("sample", material, 5.125, 200_000., .75, 5., "primary", 2)
    flight = ElasticMaterialFlight((0., 0., 0.), (0.1, 0.2, 5.), "sample", material.key,
                                   "primary", 2, 200_000., .75)
    event = ElasticScatterEvent((.1, .2, 3.), "sample", material.key, 14, .01, .02)
    trajectory = ElasticTrajectory(np.array([[0., 0., 0.], [.1, .2, 5.]]), (event,),
                                   "forward", 5.125, 2, .75, (0., 0., 1.), 200_000.)
    terminal = ElasticTerminalBundle(
        np.array([2], dtype=np.int64), np.array([[.1, .2, 5.]]), np.array([[0., 0., 1.]]),
        np.array([200_000.]), np.array([.75]), ("forward",), np.array([1], dtype=np.int64),
        np.array([True]), np.array([np.nan]), np.array([5.125]),
    )
    elastic = ElasticTransportResult((track,), (trajectory,),
                                    {"scope": "archive fixture", "scalar": np.asarray(7)},
                                    (flight,), terminal)
    vacancy = EDSVacancySignal(
        "vacancy-0", 0, 2, "sample", material.key, 14, "K", 1840., "primary",
        200_000., .75, 5.125, 1.e-22, .01, 10., .25, .75, 0.,
    )
    line = EDSLineSignal("vacancy-0", "sample", material.key, 14, "K", "Ka1", 1740.,
                         1.e-22, .01, 10., .25, .8, .9, 2., .5, (.2, .3), "primary")
    quadrature = EDSMaterialQuadrature((track,), (flight,), (2,), (2,), {"scope": "archive fixture"})
    photon = EDSPhotonRay("photon-0", (.01, -.02, 1.25), (.1, .2, .3), 1740., .5,
                          "sample", "Ka1", "vacancy-0:Ka1")
    interval = PhotonMaterialInterval("sample", 0., 0.005, material, False,
                                      "fixture", "archive fixture")
    path = EDSPhotonPathResult(photon, 1, (.1, .2, 1.5), (interval,), .9, .45,
                              "detected", "fixture")
    transport = EDSPhotonTransportResult((path,), (.2, .3), (.1, .15), True,
                                         {"scope": "archive fixture"},
                                         {"vacancy-0:Ka1": (.2, .3)}, {"vacancy-0:Ka1": .5})
    spectrum = EDSSpectrum(np.array([1735., 1745.]), np.array([.2, .3]),
                           np.array([0, 1], dtype=np.int64), (line,),
                           {"incident_electrons": 1000., "unknown_clock": float("nan")},
                           (vacancy,), elastic, transport, quadrature)
    return MaterialSectionCache({"eds": "eds-fixture", "section_material": "material-fixture"},
                                "executed-incident-fixture", (0., 0.), elastic, None,
                                target_z_mm=1234., eds_spectrum=spectrum)


def _encoded(value):
    arrays = {}
    return _pack(value, arrays, {}, _data_classes()), arrays


@pytest.fixture(scope="module")
def snapshot():
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    return capture_instrument_snapshot(default_state())


def _package(material_cache, snapshot):
    tree, arrays = _encoded(material_cache)
    return WorkingPointCheckpoint(snapshot, arrays, 1234., "archive-fixture",
                                  {"scope": "storage only", "tree": tree})


def test_complete_eds_disk_roundtrip_retains_records_arrays_and_shared_transport(material_cache, snapshot, tmp_path):
    before = _package(material_cache, snapshot)
    path = tmp_path / "eds.temwp"
    before.write_package(path)
    after = WorkingPointCheckpoint.read_package(path)
    restored = _unpack(thaw_json(after.metadata)["tree"], after.arrays, _data_classes())
    assert after.digest == before.digest
    assert _same_executed_record(restored, material_cache)
    assert restored.eds_spectrum.elastic_transport is restored.elastic_transport
    assert restored.eds_spectrum.photon_transport.paths[0].photon.direction == (
        material_cache.eds_spectrum.photon_transport.paths[0].photon.direction)
    for name, original in before.arrays.items():
        assert after.arrays[name].dtype == original.dtype
        assert after.arrays[name].tobytes() == original.tobytes()
        assert not after.arrays[name].flags.writeable


def test_current_material_record_requires_complete_fields(material_cache):
    tree, arrays = _encoded(material_cache)
    del tree["fields"]["eds_spectrum"]
    with pytest.raises(ValueError, match="missing current"):
        _unpack(tree, arrays, _data_classes())


@pytest.mark.parametrize("corruption", ["unknown_type", "unknown_field", "missing_vacancy"])
def test_invalid_nested_eds_records_are_rejected(material_cache, corruption):
    tree, arrays = _encoded(material_cache)
    eds = tree["fields"]["eds_spectrum"]
    if corruption == "unknown_type":
        eds["fields"]["photon_transport"]["type"] = "untrusted.imported.Class"
    elif corruption == "unknown_field":
        eds["fields"]["arbitrary_code"] = "never imported or executed"
    else:
        eds["fields"]["vacancies"] = {"tuple": []}
    with pytest.raises(ValueError, match="Unknown|vacancy"):
        _unpack(tree, arrays, _data_classes())


def test_eds_rejects_a_different_executed_material_track(material_cache):
    from copy import deepcopy
    from dataclasses import replace
    separate = deepcopy(material_cache.elastic_transport)
    material_cache = replace(material_cache, eds_spectrum=replace(
        material_cache.eds_spectrum, elastic_transport=separate))
    tree, arrays = _encoded(material_cache)
    elastic = tree["fields"]["eds_spectrum"]["fields"]["elastic_transport"]
    elastic["fields"]["eds_tracks"]["tuple"][0]["fields"]["path_length_nm"] += .1
    with pytest.raises(ValueError, match="share one executed elastic"):
        _unpack(tree, arrays, _data_classes())


def test_matching_summary_does_not_hide_changed_terminal_array(material_cache):
    from copy import deepcopy
    from dataclasses import replace
    separate = deepcopy(material_cache.elastic_transport)
    material_cache = replace(material_cache, eds_spectrum=replace(
        material_cache.eds_spectrum, elastic_transport=separate))
    tree, arrays = _encoded(material_cache)
    elastic = tree["fields"]["eds_spectrum"]["fields"]["elastic_transport"]
    terminal = elastic["fields"]["terminal_electrons"]["fields"]
    original_key = terminal["position_nm"]["array"]
    arrays["different-position"] = arrays[original_key].copy()
    arrays["different-position"][0, 0] += .125
    terminal["position_nm"] = {"array": "different-position"}
    with pytest.raises(ValueError, match="share one executed elastic"):
        _unpack(tree, arrays, _data_classes())


def test_archive_does_not_normalise_an_invalid_stored_photon_direction(material_cache):
    tree, arrays = _encoded(material_cache)
    photon = tree["fields"]["eds_spectrum"]["fields"]["photon_transport"]["fields"]["paths"]["tuple"][0]["fields"]["photon"]
    photon["fields"]["direction"] = {"tuple": [0., 0., 2.]}
    with pytest.raises(ValueError, match="unit vector"):
        _unpack(tree, arrays, _data_classes())


def test_changed_spectrum_array_bytes_fail_archive_integrity(material_cache, snapshot, tmp_path):
    checkpoint = _package(material_cache, snapshot)
    original = tmp_path / "original.temwp"
    broken = tmp_path / "broken.temwp"
    checkpoint.write_package(original)
    with ZipFile(original) as source, ZipFile(broken, "w") as destination:
        manifest = json.loads(source.read("manifest.json"))
        tree = thaw_json(checkpoint.metadata)["tree"]
        key = tree["fields"]["eds_spectrum"]["fields"]["expected_counts"]["array"]
        target_entry = manifest["arrays"][key]["entry"]
        for item in source.infolist():
            content = source.read(item.filename)
            if item.filename == target_entry:
                content = content[:-1] + bytes((content[-1] ^ 1,))
            destination.writestr(item.filename, content)
    with pytest.raises(ValueError, match="checksum"):
        WorkingPointCheckpoint.read_package(broken)


def _with_response(material_cache):
    from dataclasses import replace
    from temsim.detector.eds_response import capture_eds_response
    old = material_cache.eds_spectrum
    spectrum = replace(old, metrics={**old.metrics,
        "signal_kind": "characteristic_x_ray", "ionisation_model": "storage fixture",
        "detector_segment_count": 2})
    spectrum = replace(spectrum, response_rates=capture_eds_response(spectrum))
    return replace(material_cache, eds_spectrum=spectrum)


def test_response_coefficients_roundtrip_exact_and_immutable(material_cache, snapshot, tmp_path):
    cache = _with_response(material_cache)
    before = _package(cache, snapshot)
    path = tmp_path / "eds-response.temwp"
    before.write_package(path)
    after = WorkingPointCheckpoint.read_package(path)
    restored = _unpack(thaw_json(after.metadata)["tree"], after.arrays, _data_classes())
    assert _same_executed_record(restored, cache)
    with pytest.raises(ValueError):
        restored.eds_spectrum.response_rates.line_expected.setflags(write=True)


def test_response_coefficients_must_belong_to_the_stored_lines(material_cache):
    tree, arrays = _encoded(_with_response(material_cache))
    rates = tree["fields"]["eds_spectrum"]["fields"]["response_rates"]
    rates["fields"]["line_ids"]["tuple"][0]["tuple"][1] = "different-transition"
    with pytest.raises(ValueError, match="identities"):
        _unpack(tree, arrays, _data_classes())


def test_current_spectrum_requires_explicit_response_field(material_cache):
    tree, arrays = _encoded(_with_response(material_cache))
    del tree["fields"]["eds_spectrum"]["fields"]["response_rates"]
    with pytest.raises(ValueError, match="missing current"):
        _unpack(tree, arrays, _data_classes())


def test_response_mapping_order_survives_sorted_json_disk_roundtrip(material_cache, snapshot, tmp_path):
    """Mapping insertion order is not the rate-array axis: its key tuple is."""
    from dataclasses import replace
    from temsim.detector.eds_response import capture_eds_response, replay_eds_response
    spectrum = _with_response(material_cache).eds_spectrum
    photons = spectrum.photon_transport
    photons = replace(photons,
        expected_detected_weight_per_emission={"vacancy-0:Ka2": (.4, .6), "vacancy-0:Ka1": (.2, .3)},
        quadrature_weight_per_emission={"vacancy-0:Ka2": 1., "vacancy-0:Ka1": .5},
        metrics={**photons.metrics, "blocked_input_weight_by_component": {"z-holder": .25, "a-holder": .5},
                 "total_detected_weight": 1.5, "total_quadrature_weight": 1.5})
    spectrum = replace(spectrum, photon_transport=photons)
    spectrum = replace(spectrum, response_rates=capture_eds_response(spectrum))
    material_cache = replace(material_cache, eds_spectrum=spectrum)
    package = _package(material_cache, snapshot)
    path = tmp_path / "mapping-order.temwp"
    package.write_package(path)
    loaded = WorkingPointCheckpoint.read_package(path)
    restored = _unpack(thaw_json(loaded.metadata)["tree"], loaded.arrays, _data_classes()).eds_spectrum
    assert tuple(restored.photon_transport.quadrature_weight_per_emission) != spectrum.response_rates.quadrature_emission_keys
    assert _same_executed_record(restored, spectrum)
    doubled = replay_eds_response(restored, incident_electrons=2000., energy_max_ev=2000.,
        energy_bin_width_ev=10., energy_resolution_fwhm_ev=0., poisson_enabled=False, poisson_seed=0)
    for key, weight in photons.quadrature_weight_per_emission.items():
        assert doubled.photon_transport.quadrature_weight_per_emission[key] == 2.*weight
    for key, values in photons.expected_detected_weight_per_emission.items():
        np.testing.assert_array_equal(doubled.photon_transport.expected_detected_weight_per_emission[key], 2.*np.array(values))
    assert doubled.photon_transport.metrics["blocked_input_weight_by_component"] == {"a-holder": 1., "z-holder": .5}


def test_actual_tip_origin_eds_archive_resumes_without_gun_material_or_xray_reexecution(tmp_path, monkeypatch):
    """Nine physical tip rays exercise the archive and actual continuation chain."""
    from dataclasses import replace
    from threading import Event, Timer
    from temsim import simulation_pipeline as pipeline
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.cpu_resources import numerical_job
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    from temsim.particle_section_io import load_section_result, save_section_result
    from temsim.physics.particle_sections import section_limits

    state = default_state()
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), recording=next(
        item.name for item in catalog.recording_systems if not bool(item.properties.get("energy_filter"))))
    catalog.apply(state, selection)
    state.electron_gun.emitter.ray_count = 9
    state.acceleration_backend = "CPU"
    state._tuning_quality = "High accuracy"
    state.sample.inserted = True
    state.sample.specimen_mode = "reference"
    state.sample.reference_sample_key = state.sample.specimen_preset_key = "si_110"
    state.sample.thickness_nm = 5.
    state.sample.size_x_nm = state.sample.size_y_nm = 100_000.
    state.sample.centre_x_nm = state.sample.centre_y_nm = 0.
    state.sample.eds_enabled = True
    state.sample.wave_enabled = state.sample.stem_wave_enabled = False
    state.ac_deflector.scan_enabled = state.descan_deflector.scan_enabled = False
    state.vacuum_map.enabled = False
    assert not state.energy_filter_installed and not state.energy_filter.enabled
    assert state.electron_gun.emitter.coherence is None
    # Both executions consume the same detached input representation. This
    # prevents a loader/default representation difference becoming the test.
    state = capture_instrument_snapshot(state).restore()
    cancel = Event()
    deadline = Timer(120., cancel.set)
    deadline.daemon = True
    state._tuning_cancelled = cancel.is_set
    target = section_limits(state)[1]
    deadline.start()
    try:
        with numerical_job(4, cancelled=cancel.is_set):
            first = pipeline.calculate_particle_section(state, target)
            spectrum = first.specimen_interactions.eds_spectrum
            assert spectrum.total_expected_counts > 0.
            assert first.specimen_interactions.elastic_transport.metrics["material_hit_trajectory_count"] > 0
            assert len(first.simulation.gun_trace.exit_bundle.ray_id) == 9
            assert first.specimen_exit.metrics["source_probability_conserved"]
            path = tmp_path / "physical-eds.temsection"
            save_section_result(first, path)
            restored = load_section_result(path)
            saved_cache = restored.simulation.material_section_cache
            assert restored.section_archive_info["eds_completed"] is True
            assert _same_executed_record(saved_cache.eds_spectrum, spectrum)
            assert saved_cache.eds_spectrum.elastic_transport is saved_cache.elastic_transport

            def forbidden(*args, **kwargs):
                pytest.fail("A compatible archived upstream/EDS calculation must not execute again")

            monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", forbidden)
            monkeypatch.setattr("temsim.specimen.elastic_transport.simulate_elastic_point_transport", forbidden)
            monkeypatch.setattr("temsim.detector.eds_signal.simulate_eds_point", forbidden)
            monkeypatch.setattr("temsim.detector.eds_signal.simulate_eds_tracks", forbidden)
            changed = restored.state_snapshot
            changed._tuning_cancelled = cancel.is_set
            projection = next(lens for lens in changed.lenses if lens.key == "projector_lens_2")
            projection.percent += .1
            continued = pipeline.calculate_particle_section(
                changed, target, ("projector_lens_2",), existing_result=restored)
            assert {"elastic", "inelastic", "eds"} <= continued.reused_products
            assert "eds" not in continued.calculated_products
            assert continued.specimen_interactions.eds_spectrum is saved_cache.eds_spectrum
            assert continued.simulation.metrics["section_gun_reused"]
            assert continued.specimen_exit.metrics["source_probability_conserved"]
            np.testing.assert_array_equal(continued.specimen_interactions.eds_spectrum.expected_counts,
                                          spectrum.expected_counts)
    finally:
        deadline.cancel()
