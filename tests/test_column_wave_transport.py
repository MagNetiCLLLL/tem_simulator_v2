from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from temsim.optics.column import default_state
from temsim.physics.column_wave import _linear_factor, _propagate_column
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.specimen_wave_transport import _regrid_mode, _slice_phase
from test_wave_detector_readout import checkpoint
from temsim.physics.wave_reference import AxialWaveReference


def test_constant_hamiltonian_lift_matches_matrix_exponential():
    g = 1.2
    r = np.array(((0, g), (-g, 0)))
    a = np.block([[r, np.eye(2)], [-np.eye(2)*3., r]])
    path = CanonicalPath(1e-3)
    _linear_factor(path, a, .04)
    np.testing.assert_allclose(path.matrix, expm(a*.04), atol=1e-14)


def test_column_vacuum_transports_actual_shape_energy_and_weight(monkeypatch):
    """Independent analytic vacuum operator; not a captured-instrument chain."""
    from test_tip_wave_pipeline import _quiet_prepared_column
    monkeypatch.setattr("temsim.physics.column_wave._prepare_column", _quiet_prepared_column)
    state = default_state()
    for collection in (state.lenses, state.stigmators, state.corrector_elements, state.deflectors):
        for component in collection:
            component.enabled = False
    state.apertures = []
    c = checkpoint(two=True)
    references = (AxialWaveReference(2e-8, 1e-22), AxialWaveReference(3e-8, 2e-22))
    c = replace(c, beam=replace(c.beam, modes=tuple(replace(m, axial_reference=r) for m, r in zip(c.beam.modes, references))))
    result = _propagate_column(state, c, 2001., maximum_step_mm=.1)
    assert result.plane_z_mm == 2001.
    assert result.beam.total_weight == pytest.approx(c.beam.total_weight, rel=1e-10)
    assert [m.mode_id for m in result.beam.modes] == [m.mode_id for m in c.beam.modes]
    assert result.record["upstream_digest"] == c.digest
    from temsim.physics.tip_gun_wave import _momentum_velocity
    momentum, velocity = _momentum_velocity(300000.)
    for old, new in zip(c.beam.modes, result.beam.modes):
        assert new.axial_reference.flight_time_s == pytest.approx(old.axial_reference.flight_time_s+.001/velocity, rel=1e-14)
        assert new.axial_reference.longitudinal_action_j_s == pytest.approx(old.axial_reference.longitudinal_action_j_s+.001*momentum, rel=1e-14, abs=1e-40)
    assert not np.array_equal(result.beam.modes[0].plane.amplitude, c.beam.modes[0].plane.amplitude)


def test_captured_residual_electric_field_is_executed_in_column_wave_operator():
    """Actual captured scalar field; isolated downstream E, not a full chain."""
    state = default_state()
    for collection in (state.lenses, state.stigmators, state.corrector_elements, state.deflectors):
        for component in collection:
            component.enabled = False
    state.apertures = []
    from temsim.physics.column_wave import _prepare_column
    prepared = _prepare_column(state, 2000., 2001., .1)
    field = prepared[0].electric_field
    potentials = field.potential_rise_v_at_global_positions(np.array(((0., 0., 2.), (0., 0., 2.001))))
    result = _propagate_column(state, checkpoint(), 2001., maximum_step_mm=.1, _prepared=prepared)
    assert result.beam.modes[0].energy_kev == pytest.approx(300.+(potentials[-1]-potentials[0])*.001, abs=1e-12)
    assert "captured static scalar field" in result.record["electric_model"]
    assert result.beam.total_weight == pytest.approx(checkpoint().beam.total_weight, rel=1e-10)


def test_segmented_column_uses_actual_slotted_propagation_plan(tmp_path, monkeypatch):
    """Exercise production plan storage, not only analytic namespace fixtures."""
    import temsim.physics.column_wave as column
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    state = default_state()
    prepared = column._prepare_column(state, 2000., 2000.01, .01)
    assert not hasattr(prepared[0], "__dict__")
    source = checkpoint(two=True)
    monkeypatch.setattr(column, "numerical_thread_budget", lambda requested=None: 2)
    output, hit = column._propagate_column_segmented(state, source, 2000.01,
        store=ExecutedWaveStore(tmp_path, "actual-slotted-column", 1<<28),
        maximum_step_mm=.01, segment_steps=128, _prepared=prepared)
    assert not hit
    assert output.plane_z_mm == 2000.01
    assert [m.mode_id for m in output.beam.modes] == [m.mode_id for m in source.beam.modes]
    assert output.beam.total_weight == pytest.approx(source.beam.total_weight, rel=1e-10)
    resources = output.record["mode_execution"]
    assert resources["workers"] == 2
    assert resources["shared_retained_bytes"] > 0
    assert (resources["shared_retained_bytes"] + resources["workers"]*resources["worker_working_bytes"]
            <= resources["maximum_working_bytes"])


def test_regrid_retains_phase_carriers_without_fitting_or_cropping():
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    mode = checkpoint().beam.modes[0]
    axis = (np.arange(128)-64)*.51e-6
    result, record = _regrid_mode(mode, axis, axis)
    assert result.weight_per_reference_electron == mode.weight_per_reference_electron
    assert record["interpolation_difference"] < 1e-4
    assert result.plane.probability == pytest.approx(1., abs=1e-12)
    original_phase = np.angle(mode.plane.full_amplitude(float(wavelength_m(300000)))[32, 32])
    new_phase = np.angle(result.plane.full_amplitude(float(wavelength_m(300000)))[64, 64])
    assert new_phase == pytest.approx(original_phase, abs=1e-12)
    with pytest.raises(ValueError, match="crop"):
        _regrid_mode(mode, axis/2, axis/2)


def test_specimen_phase_is_complex_and_not_an_intensity_change():
    mode = checkpoint().beam.modes[0]
    axis = (np.arange(64)-32)*1e-6
    potential = np.ones((64, 64))*3.
    result = _slice_phase(mode, potential, axis, axis, .02, .5)
    np.testing.assert_allclose(result.plane.amplitude, mode.plane.amplitude*np.exp(.03j), atol=1e-14)
    assert result.weight_per_reference_electron == mode.weight_per_reference_electron
    assert result.plane.probability == pytest.approx(1.)


def _off_axis_affine_clip_wave(shape=(5, 7)):
    from temsim.physics.multiplane_wave import PlaneWave
    iy, ix = np.mgrid[:shape[0], :shape[1]]
    amplitude = (1+ix+2*iy)*np.exp(1j*(.13*ix-.21*iy))
    amplitude /= np.linalg.norm(amplitude)
    return PlaneWave(amplitude, np.array(((1.2e-6, .6e-6), (-.4e-6, .9e-6))),
        np.array((1.3e-6, -2.1e-6)), np.array(((1., 2.), (2., 3.))), np.array((1e-4, -2e-4)))


@pytest.mark.parametrize("radius", (.1, np.inf, -np.inf, np.nan))
@pytest.mark.parametrize("shape", ((5, 7), (6, 8)))
def test_clear_column_bore_does_not_allocate_coordinates_or_scan_amplitudes(monkeypatch, radius, shape):
    from temsim.physics.column_wave import _clip
    from temsim.physics.multiplane_wave import PlaneWave
    wave = _off_axis_affine_clip_wave(shape)
    monkeypatch.setattr(PlaneWave, "coordinates_m", lambda *_a:
                        pytest.fail("A provably clear bore needs no full coordinate grid"))
    monkeypatch.setattr(PlaneWave, "probability", property(lambda _self:
                        pytest.fail("A provably clear bore needs no amplitude scan")))
    result, rows = _clip(wave, radius, (), 42., .3)
    assert result is wave
    assert rows == []


@pytest.mark.parametrize("boundary", ("inside", "cut", "equal", "one_ulp_inside", "zero", "negative"))
def test_column_bore_fast_proof_preserves_pointwise_mask_phase_and_loss_ledger(monkeypatch, boundary):
    from temsim.physics.column_wave import _clip
    from temsim.physics.multiplane_wave import PlaneWave
    wave = _off_axis_affine_clip_wave()
    xy = wave.coordinates_m()*1e3
    distances = np.hypot(xy[0], xy[1])
    outer = float(distances.max())
    radius = {"inside": outer*2, "cut": float(np.median(distances)), "equal": outer,
              "one_ulp_inside": np.nextafter(outer, np.inf), "zero": 0., "negative": -.1}[boundary]
    mask = distances < radius  # The physical bore uses a strict boundary.
    expected = np.where(mask, wave.amplitude, 0j)
    lost = wave.probability-float(np.sum(abs(expected)**2))
    expected_rows = ([{"component": "column_wall", "z_mm": 42., "lost_weight": .3*lost}]
                     if lost > 0 else [])
    coordinate_calls = []
    original_coordinates = PlaneWave.coordinates_m
    def coordinates(current):
        coordinate_calls.append(current)
        return original_coordinates(current)
    monkeypatch.setattr(PlaneWave, "coordinates_m", coordinates)
    result, rows = _clip(wave, radius, (), 42., .3)
    np.testing.assert_array_equal(result.amplitude, expected)
    np.testing.assert_array_equal(result.basis_m, wave.basis_m)
    np.testing.assert_array_equal(result.origin_m, wave.origin_m)
    np.testing.assert_array_equal(result.curvature_m1, wave.curvature_m1)
    np.testing.assert_array_equal(result.tilt_rad, wave.tilt_rad)
    assert rows == expected_rows
    # A strict/effectively touching boundary uses the original full mask even
    # when the radius is one ULP larger. The proof does not round it inward.
    assert len(coordinate_calls) == (0 if boundary == "inside" else 1)


@pytest.mark.parametrize("radius", (.1, np.inf, np.nan))
def test_clear_bore_still_executes_custom_aperture_and_exact_loss_ledger(radius):
    from types import SimpleNamespace
    from temsim.physics.column_wave import _clip
    wave = _off_axis_affine_clip_wave()
    xy = wave.coordinates_m()*1e3
    seen = []
    def transmission(x, y):
        seen.append((x.copy(), y.copy()))
        return (x >= .001) & (y <= 0.)
    aperture = SimpleNamespace(key="custom-aperture", radius_mm=1., transmission_mask=transmission)
    result, rows = _clip(wave, radius, (aperture,), 42., .3)
    expected = np.where((xy[0] >= .001) & (xy[1] <= 0.), wave.amplitude, 0j)
    before, after = wave.probability, float(np.sum(abs(expected)**2))
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], xy)
    np.testing.assert_array_equal(result.amplitude, expected)
    assert rows == [{"component": aperture.key, "z_mm": 42., "input_weight": .3*before,
                     "output_weight": .3*after, "lost_weight": .3*(before-after)}]


def test_column_bore_and_aperture_losses_remain_ordered_and_separate():
    from types import SimpleNamespace
    from temsim.physics.column_wave import _clip
    wave = _off_axis_affine_clip_wave()
    xy = wave.coordinates_m()*1e3
    radius = float(np.median(np.hypot(xy[0], xy[1])))
    aperture = SimpleNamespace(key="offset-aperture", radius_mm=.002,
                               offset_x_mm=.001, offset_y_mm=-.001)
    inside_wall = np.hypot(xy[0], xy[1]) < radius
    inside_aperture = np.hypot(xy[0]-aperture.offset_x_mm, xy[1]-aperture.offset_y_mm) <= aperture.radius_mm
    after_wall = np.where(inside_wall, wave.amplitude, 0j)
    expected = np.where(inside_aperture, after_wall, 0j)
    initial = wave.probability
    middle, final = (float(np.sum(abs(value)**2)) for value in (after_wall, expected))
    result, rows = _clip(wave, radius, (aperture,), 42., .3)
    np.testing.assert_array_equal(result.amplitude, expected)
    assert rows == [{"component": "column_wall", "z_mm": 42., "lost_weight": .3*(initial-middle)},
                    {"component": aperture.key, "z_mm": 42., "input_weight": .3*middle,
                     "output_weight": .3*final, "lost_weight": .3*(middle-final)}]


def test_clear_bore_keeps_invalid_aperture_mask_rejection():
    from types import SimpleNamespace
    from temsim.physics.column_wave import _clip
    aperture = SimpleNamespace(key="invalid-aperture", radius_mm=1.,
                               transmission_mask=lambda _x, _y: np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="wrong shape"):
        _clip(_off_axis_affine_clip_wave(), .1, (aperture,), 42., .3)


def _quadratic_fixture(count=6):
    """Small analytic plan; no claim to qualify the captured whole instrument."""
    from types import SimpleNamespace
    z = 2000.+np.arange(count+1)*.2
    plan = SimpleNamespace(z_mm=z, step_m=np.diff(z)*1e-3,
        midpoint_magnetic_t=np.zeros(count), midpoint_sx_m2=np.zeros(count),
        midpoint_sy_m2=np.zeros(count), midpoint_sxy_m2=np.zeros(count),
        midpoint_hex_normal_m3=np.zeros(count), midpoint_hex_skew_m3=np.zeros(count),
        kick_x_rad=np.zeros(count+1), kick_y_rad=np.zeros(count+1),
        cs_kick_m3=np.zeros(count+1), electric_field=None,
        signature=f"independent-quadratic-test:{count}")
    return plan, np.full(count+1, np.inf), {}, []


def _resolved_non_gaussian_wave():
    from temsim.physics.multiplane_wave import PlaneWave
    yy, xx = np.mgrid[-32:32, -32:32]
    envelope = np.exp(-(xx**2+yy**2)/50.)*(1+.2j*xx/5.+.15*np.cos(.7*yy/5.))
    envelope /= np.linalg.norm(envelope)
    return PlaneWave(envelope, np.array(((2.5e-7, .3e-7), (-.2e-7, 2.2e-7))),
                     np.array((.3e-6, -.2e-6)), np.array(((.01, .003), (.003, -.006))),
                     np.array((.8e-7, -.5e-7)))


def _drift_path(distance):
    path = CanonicalPath(1e-3)
    matrix = np.eye(4)
    matrix[:2, 2:] = np.eye(2)*distance
    path.append(matrix)
    return path


def _run_candidate(wave, paths, prepared, wavelength=2e-12):
    from temsim.physics.column_wave import _linear_run
    plan, radii, stops, _ = prepared
    return _linear_run(wave, wavelength, paths, 0, plan, radii, stops,
                       plan.kick_x_rad, plan.kick_y_rad)


def test_grouped_constant_force_has_independent_schrodinger_phase_and_displacement():
    """Uniform force: psi=e^(ik(F.x L-F²L³/6)) free(psi)(x-FL²/2)."""
    from temsim.physics.multiplane_wave import propagate_plane_wave
    wave = replace(_resolved_non_gaussian_wave(), origin_m=np.zeros(2),
                   tilt_rad=np.zeros(2), curvature_m1=np.zeros((2, 2)))
    prepared = _quadratic_fixture()
    plan = prepared[0]
    force = np.array((3e-4, -2e-4))
    generator = np.block([[np.zeros((2, 2)), np.eye(2)], [np.zeros((2, 2)), np.zeros((2, 2))]])
    paths = []
    for distance in plan.step_m:
        path = CanonicalPath(1e-3)
        _linear_factor(path, generator, float(distance), force=np.r_[np.zeros(2), force])
        paths.append(path)
    last, combined = _run_candidate(wave, paths, prepared)
    assert last == len(paths)
    distance = float(plan.step_m.sum())
    np.testing.assert_allclose(combined.offset, np.r_[force*distance**2/2, force*distance], rtol=2e-13, atol=1e-22)
    assert combined.action_m == pytest.approx(float(force@force)*distance**3/12, rel=2e-13, abs=1e-30)
    wavelength = 2e-12
    result = propagate_plane_wave(wave, combined.matrix, combined.offset, wavelength, **combined.phase_kwargs())
    fy, fx = np.meshgrid(np.fft.fftfreq(64), np.fft.fftfreq(64), indexing="ij")
    frequencies = np.einsum("ij,jyx->iyx", np.linalg.inv(wave.basis_m).T, np.stack((fx, fy)))
    free = np.fft.ifft2(np.fft.fft2(wave.amplitude)*np.exp(-1j*np.pi*wavelength*distance*np.sum(frequencies**2, axis=0)))
    # At the moving origin x=F L²/2 the exact force phase is k F²L³/3.
    expected = free*np.exp(2j*np.pi/wavelength*float(force@force)*distance**3/3)
    np.testing.assert_allclose(result.amplitude, expected, rtol=2e-11, atol=2e-13)
    np.testing.assert_allclose(result.origin_m, force*distance**2/2, rtol=2e-13, atol=1e-22)
    np.testing.assert_allclose(result.tilt_rad, force*distance, rtol=2e-13, atol=1e-22)
    assert result.probability == pytest.approx(wave.probability, abs=1e-13)


def test_grouped_noncommuting_paths_retain_absolute_complex_field_and_affine_action():
    from temsim.physics.multiplane_wave import propagate_plane_wave
    wave, prepared = _resolved_non_gaussian_wave(), _quadratic_fixture()
    paths = []
    for i, distance in enumerate(prepared[0].step_m):
        rotation = np.array(((0., .4*(-1)**i), (-.4*(-1)**i, 0.)))
        stiffness = np.array(((2.+i, .7), (.7, 4.-i*.2)))
        generator = np.block([[rotation, np.eye(2)], [-stiffness, rotation]])
        path = CanonicalPath(1e-3)
        _linear_factor(path, generator, float(distance), force=np.array((0., 0., 3e-4, -2e-4)))
        paths.append(path)
    assert np.linalg.norm(paths[0].matrix@paths[1].matrix-paths[1].matrix@paths[0].matrix) > 1e-10
    last, combined = _run_candidate(wave, paths, prepared)
    assert last == len(paths)
    sequential = wave
    for path in paths:
        sequential = propagate_plane_wave(sequential, path.matrix, path.offset, 2e-12, **path.phase_kwargs())
    grouped = propagate_plane_wave(wave, combined.matrix, combined.offset, 2e-12, **combined.phase_kwargs())
    np.testing.assert_allclose(grouped.basis_m, sequential.basis_m, rtol=2e-12, atol=1e-20)
    np.testing.assert_allclose(grouped.origin_m, sequential.origin_m, rtol=2e-12, atol=1e-20)
    np.testing.assert_allclose(grouped.curvature_m1, sequential.curvature_m1, rtol=2e-11, atol=1e-13)
    np.testing.assert_allclose(grouped.full_amplitude(2e-12), sequential.full_amplitude(2e-12), rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("event", ("aperture", "kick_x", "kick_y", "spherical", "hex_normal", "hex_skew", "wall"))
def test_grouping_stops_before_every_intermediate_hardware_or_nonlinear_operation(event):
    wave, prepared = _resolved_non_gaussian_wave(), _quadratic_fixture()
    plan, radii, stops, _ = prepared
    node = 3
    if event == "aperture":
        stops[node] = (object(),)
    elif event.startswith("kick_"):
        getattr(plan, event+"_rad")[node] = 1e-7
    elif event == "spherical":
        plan.cs_kick_m3[node] = 1e3
    elif event.startswith("hex_"):
        setattr(plan, "midpoint_"+event+"_m3", np.eye(1, len(plan.step_m), node-1).ravel())
    else:
        radii[node] = 1e-4
    last, combined = _run_candidate(wave, [_drift_path(dz) for dz in plan.step_m], prepared)
    assert last == node-1
    assert combined is not None
    assert combined.matrix[0, 2] == pytest.approx(float(plan.step_m[:node-1].sum()), abs=1e-16)


def test_grouping_rejects_child_internal_winding_even_with_identity_endpoint():
    wave, prepared = _resolved_non_gaussian_wave(), _quadratic_fixture(1)
    generator = np.zeros((4, 4))
    generator[0, 2], generator[2, 0] = 1., -1.
    child = CanonicalPath(1e-3)
    # Resolve the executed cycle, instead of inferring its branch from the
    # identity endpoint (which cannot distinguish any number of full turns).
    for _ in range(64):
        _linear_factor(child, generator, 2*np.pi/64)
    np.testing.assert_allclose(child.matrix, np.eye(4), atol=1e-13)
    # One full oscillator cycle in one coordinate has the nontrivial lift -I.
    assert np.exp(1j*child.reference_phase_rad) == pytest.approx(-1.+0j, abs=1e-12)
    assert _run_candidate(wave, [child], prepared) == (0, None)


def test_grouping_preserves_last_valid_path_when_cumulative_lift_is_undersampled():
    wave, prepared = _resolved_non_gaussian_wave(), _quadratic_fixture(2)
    lens = CanonicalPath(1e-3)
    matrix = np.eye(4)
    matrix[2, 0] = -2000.
    lens.append(matrix)
    last, combined = _run_candidate(wave, [lens, _drift_path(.001)], prepared)
    assert last == 1
    np.testing.assert_array_equal(combined.matrix, lens.matrix)
    assert combined.reference_phase_rad == lens.reference_phase_rad


def test_grouping_rejects_nearly_singular_carrier_chart():
    wave, prepared = _resolved_non_gaussian_wave(), _quadratic_fixture(1)
    child = CanonicalPath(1e-3)
    child.append(np.diag((1e-9, 1., 1e9, 1.)))
    assert _run_candidate(wave, [child], prepared) == (0, None)


@pytest.mark.parametrize("distance, accepted", ((15e-6, True), (16e-6, False), (17e-6, False)))
def test_grouping_full_band_bound_uses_each_axis_of_an_anisotropic_lattice(distance, accepted):
    wave = replace(_resolved_non_gaussian_wave(), basis_m=np.diag((1e-9, 1e-6)),
                   curvature_m1=np.zeros((2, 2)))
    prepared = _quadratic_fixture(1)
    last, combined = _run_candidate(wave, [_drift_path(distance)], prepared)
    # At lambda=2 pm the most oblique represented x frequency travels half
    # lambda*L/dx² cells. The guard is strict at one quarter of 64 cells.
    assert (last == 1) is accepted
    assert (combined is not None) is accepted


def _sample_full_field_at_points(wave, wavelength, points):
    """Independent Fourier-series evaluation on shared physical coordinates."""
    ny, nx = wave.amplitude.shape
    local = points-wave.origin_m[:, None]
    indices = np.linalg.solve(wave.basis_m, local)+np.array((nx//2, ny//2))[:, None]
    fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    phase = fx.ravel()[:, None]*indices[0]+fy.ravel()[:, None]*indices[1]
    sampled = np.fft.fft2(wave.amplitude).ravel()@np.exp(2j*np.pi*phase)/(nx*ny)
    q = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    carrier = (np.einsum("ip,ij,jp->p", local, q, local)/2+tilt@local)*2*np.pi/wavelength
    return sampled*np.exp(1j*carrier)/np.sqrt(abs(np.linalg.det(wave.basis_m)))


def test_column_grouped_and_stepwise_paths_preserve_complex_modes_and_axial_references():
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    source = checkpoint(two=True)
    plane = _resolved_non_gaussian_wave()
    modes = tuple(replace(mode, plane=replace(plane, amplitude=plane.amplitude*np.exp(.4j*i)),
                          energy_kev=300.-20*i, axial_reference=AxialWaveReference(2e-8+i*1e-9, 1e-22))
                  for i, mode in enumerate(source.beam.modes))
    source = replace(source, beam=replace(source.beam, modes=modes))
    prepared = _quadratic_fixture()
    plan = prepared[0]
    count = len(plan.step_m)
    plan.midpoint_sx_m2 = np.linspace(2., 4., count)
    plan.midpoint_sy_m2 = np.linspace(5., 3., count)
    plan.midpoint_sxy_m2 = np.full(count, .7)
    plan.midpoint_magnetic_t = np.linspace(1e-5, 2e-5, count)
    plan.dipole_bx_t = np.full(3*count, 2e-7)
    plan.dipole_by_t = np.full(3*count, -3e-7)
    state = default_state()
    fast = _propagate_column(state, source, plan.z_mm[-1], _prepared=prepared)
    reference = _propagate_column(state, source, plan.z_mm[-1], _prepared=prepared, _combine_linear=False)
    yy, xx = np.meshgrid(np.linspace(-2e-6, 2e-6, 11), np.linspace(-2e-6, 2e-6, 13), indexing="ij")
    points = np.stack((xx.ravel(), yy.ravel()))
    assert all(row["quadratic_runs"] for row in fast.record["modes"])
    for actual, expected in zip(fast.beam.modes, reference.beam.modes):
        assert actual.mode_id == expected.mode_id
        assert actual.energy_kev == expected.energy_kev
        assert actual.axial_reference == expected.axial_reference
        assert actual.weight_per_reference_electron == pytest.approx(expected.weight_per_reference_electron, abs=1e-12)
        wavelength = float(wavelength_m(actual.energy_kev*1000))
        a = _sample_full_field_at_points(actual.plane, wavelength, points)
        b = _sample_full_field_at_points(expected.plane, wavelength, points)
        # The absolute complex error includes scalar action and lift; no fitted
        # global phase, fitted beam shape, or intensity-only comparison.
        assert np.linalg.norm(a-b)/np.linalg.norm(b) < 2e-8


def test_column_grouping_keeps_actual_intermediate_and_final_aperture_losses():
    from types import SimpleNamespace
    source, prepared = checkpoint(), _quadratic_fixture()
    plan, radii, stops, _ = prepared
    # Deterministic partial masks exercise real loss accounting without a
    # special geometry approximation in the grouping planner.
    for node, offset in ((3, 0.), (len(plan.step_m), -2e-3)):
        stops[node] = (SimpleNamespace(key=f"cut:{node}", radius_mm=1.,
            transmission_mask=lambda x, y, offset=offset: x <= offset),)
    state = default_state()
    fast = _propagate_column(state, source, plan.z_mm[-1], _prepared=prepared)
    reference = _propagate_column(state, source, plan.z_mm[-1], _prepared=prepared, _combine_linear=False)
    actual_rows, expected_rows = fast.record["modes"][0]["losses"], reference.record["modes"][0]["losses"]
    assert [row["component"] for row in actual_rows] == ["cut:3", f"cut:{len(plan.step_m)}"]
    for actual, expected in zip(actual_rows, expected_rows):
        assert actual["z_mm"] == expected["z_mm"]
        for key in ("input_weight", "output_weight", "lost_weight"):
            assert actual[key] == pytest.approx(expected[key], abs=2e-10)
    assert fast.beam.total_weight == pytest.approx(reference.beam.total_weight, abs=2e-10)


def _parallel_column_fixture(count=3):
    """Different coherent states through one analytic field and physical stop."""
    from types import SimpleNamespace
    original = checkpoint()
    plane = _resolved_non_gaussian_wave()
    modes = tuple(replace(original.beam.modes[0], mode_id=f"column-mode:{i}",
        plane=replace(plane, amplitude=plane.amplitude*np.exp(.37j*i)),
        energy_kev=300.-20*i, weight_per_reference_electron=.4-.1*i,
        axial_reference=AxialWaveReference(2e-8+i*1e-9, (1+i)*1e-22))
        for i in range(count))
    original = replace(original, beam=replace(original.beam, modes=modes))
    prepared = _quadratic_fixture(4)
    plan = prepared[0]
    plan.midpoint_magnetic_t[:] = 1e-5
    plan.midpoint_sx_m2[:] = 2.
    plan.midpoint_sy_m2[:] = 3.
    plan.midpoint_sxy_m2[:] = .7
    plan.dipole_bx_t = np.full(3*len(plan.step_m), 2e-7)
    plan.dipole_by_t = np.full(3*len(plan.step_m), -3e-7)
    # The final mask removes real probability and preserves the complex phase
    # of transmitted cells. Its effect cannot be hidden by renormalizing weight.
    prepared[2][4] = (SimpleNamespace(key="offset-cut", radius_mm=1.,
        transmission_mask=lambda x, y: (x <= .001) & (y >= -.001)),)
    return original, prepared


def _track_real_column_executor(monkeypatch):
    """Observe real numerical threads; never replace a propagation operator."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import temsim.physics.column_wave as column
    observed = {"workers": [], "threads": set(), "active": 0, "peak": 0}
    lock = threading.Lock()
    class ObservedExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers=None, **kwargs):
            observed["workers"].append(max_workers)
            self._first_batch = threading.Barrier(max_workers, timeout=5)
            self._submitted = 0
            super().__init__(max_workers=max_workers, **kwargs)

        def submit(self, function, /, *args, **kwargs):
            initial = self._submitted < self._first_batch.parties
            self._submitted += 1
            def execute():
                with lock:
                    observed["threads"].add(threading.get_ident())
                    observed["active"] += 1
                    observed["peak"] = max(observed["peak"], observed["active"])
                try:
                    if initial:
                        self._first_batch.wait()
                    return function(*args, **kwargs)
                finally:
                    with lock:
                        observed["active"] -= 1
            return super().submit(execute)
    monkeypatch.setattr(column, "ThreadPoolExecutor", ObservedExecutor)
    return observed


def _assert_column_complex_result(actual, expected):
    assert actual.plane_z_mm == expected.plane_z_mm
    assert actual.reference_current_a == expected.reference_current_a
    assert [m.mode_id for m in actual.beam.modes] == [m.mode_id for m in expected.beam.modes]
    for a, b in zip(actual.beam.modes, expected.beam.modes):
        assert a.energy_kev == b.energy_kev
        assert a.axial_reference == b.axial_reference
        assert a.scattering_history == b.scattering_history
        assert a.weight_per_reference_electron == pytest.approx(b.weight_per_reference_electron, abs=1e-14)
        # Absolute complex envelope and its complete analytical phase carrier;
        # no fitted phase, rescaling, or intensity-only equality is accepted.
        for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            np.testing.assert_allclose(getattr(a.plane, name), getattr(b.plane, name),
                                       rtol=2e-13, atol=2e-15)
    assert actual.record["modes"] == expected.record["modes"]
    resources = actual.record["mode_execution"]
    assert (resources["shared_retained_bytes"] + resources["workers"]*resources["worker_working_bytes"]
            <= resources["maximum_working_bytes"])


@pytest.mark.parametrize("cpu_limit", (1, 2))
def test_segmented_column_parallel_preserves_complex_states_losses_and_cpu_limit(tmp_path, monkeypatch, cpu_limit):
    import temsim.physics.column_wave as column
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    source, prepared = _parallel_column_fixture()
    state = default_state()
    monkeypatch.setattr(column, "numerical_thread_budget", lambda requested=None: 1)
    reference, _ = column._propagate_column_segmented(state, source, prepared[0].z_mm[-1],
        store=ExecutedWaveStore(tmp_path/"serial", "same-column", 1<<28),
        segment_steps=4, _prepared=prepared)
    observed = _track_real_column_executor(monkeypatch)
    monkeypatch.setattr(column, "numerical_thread_budget", lambda requested=None: cpu_limit)
    actual, reused = column._propagate_column_segmented(state, source, prepared[0].z_mm[-1],
        store=ExecutedWaveStore(tmp_path/"parallel", "same-column", 1<<28),
        segment_steps=4, _prepared=prepared)
    assert not reused
    _assert_column_complex_result(actual, reference)
    assert 0 < actual.beam.total_weight < source.beam.total_weight
    assert all(row["losses"][0]["component"] == "offset-cut" for row in actual.record["modes"])
    assert not np.allclose(actual.beam.modes[0].plane.origin_m,
                           source.beam.modes[0].plane.origin_m, rtol=0., atol=1e-12)
    assert actual.record["mode_execution"]["workers"] <= cpu_limit
    if cpu_limit == 1:
        assert not observed["workers"]
    else:
        assert observed["workers"] == [2]
        assert len(observed["threads"]) == observed["peak"] == 2


@pytest.mark.parametrize("failure", ("cancel", "invalid_physical_aperture"))
def test_parallel_column_failure_aborts_partial_writer_without_publication(tmp_path, monkeypatch, failure):
    import threading
    from types import SimpleNamespace
    import temsim.physics.column_wave as column
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    source, prepared = _parallel_column_fixture(2)
    stored, cancelled, writers, published = threading.Event(), threading.Event(), [], []
    second = source.beam.modes[1]
    second = replace(second, plane=replace(second.plane, origin_m=np.array((1e-4, 0.))))
    source = replace(source, beam=replace(source.beam, modes=(source.beam.modes[0], second)))
    def physical_mask(x, y):
        if float(x.mean()) > .05:
            assert stored.wait(5), "First successful mode was not streamed"
            if failure == "invalid_physical_aperture":
                return np.ones((2, 2), dtype=bool)
        return x <= .001
    prepared[2][4] = (SimpleNamespace(key="controlled-stop", radius_mm=1.,
                                     transmission_mask=physical_mask),)
    store = ExecutedWaveStore(tmp_path, "no-partial-column", 1<<28)
    make_writer = store.writer
    def tracked_writer(*args):
        writer = make_writer(*args)
        writers.append(writer)
        append = writer.append
        def append_completed(mode):
            append(mode)
            if failure == "cancel":
                cancelled.set()
            stored.set()
        writer.append = append_completed
        return writer
    monkeypatch.setattr(store, "writer", tracked_writer)
    monkeypatch.setattr(column, "numerical_thread_budget", lambda requested=None: 2)
    observed = _track_real_column_executor(monkeypatch)
    exception, message = ((InterruptedError, "cancel") if failure == "cancel" else (ValueError, "wrong shape"))
    with pytest.raises(exception, match=message):
        column._propagate_column_segmented(default_state(), source, prepared[0].z_mm[-1],
            store=store, segment_steps=4, _prepared=prepared, cancelled=cancelled.is_set,
            checkpoint_callback=published.append)
    assert stored.is_set() and len(writers) == 1  # Genuine failures are never retried as memory failures.
    assert writers[0].rows and not writers[0]._published
    assert observed["workers"] == [2] and observed["active"] == 0
    assert not published and store.get(writers[0].key) is None
    assert not list(tmp_path.iterdir())


def _refining_column_modes():
    from test_column_wave_electric import _astigmatic_focus_checkpoint
    from test_tip_wave_pipeline import _quiet_prepared_column
    source = _astigmatic_focus_checkpoint(64)
    first = source.beam.modes[0]
    source = replace(source, beam=replace(source.beam, modes=(first,
        replace(first, mode_id="second-focus-mode", weight_per_reference_electron=.3, plane=replace(first.plane,
            amplitude=first.plane.amplitude*np.exp(.47j))))))
    return source, _quiet_prepared_column(None, 2000., 2000.1, .1)


def test_parallel_column_partition_memory_retries_serial_without_changing_complex_result(tmp_path, monkeypatch):
    import temsim.physics.column_wave as column
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.wave_grid import WaveGridNumerics
    source, prepared = _refining_column_modes()
    numerics = WaveGridNumerics(maximum_pixels=512, maximum_working_bytes=128*1024**2)
    monkeypatch.setattr(column, "numerical_thread_budget", lambda requested=None: 1)
    reference, _ = column._propagate_column_segmented(default_state(), source, 2000.1,
        store=ExecutedWaveStore(tmp_path/"serial", "refined-column", 1<<28), segment_steps=1,
        _prepared=prepared, grid_numerics=numerics)
    observed = _track_real_column_executor(monkeypatch)
    monkeypatch.setattr(column, "numerical_thread_budget", lambda requested=None: 2)
    actual, _ = column._propagate_column_segmented(default_state(), source, 2000.1,
        store=ExecutedWaveStore(tmp_path/"retry", "refined-column", 1<<28), segment_steps=1,
        _prepared=prepared, grid_numerics=numerics)
    _assert_column_complex_result(actual, reference)
    assert actual.record["mode_execution"]["retry_serial_for_memory"]
    assert observed["workers"] == [2] and observed["active"] == 0
    assert all(m.plane.amplitude.shape == (512, 512) for m in actual.beam.modes)
    assert len(list((tmp_path/"retry").glob("*.json"))) == 1
    assert len(list((tmp_path/"retry").iterdir())) == 2  # Published index + its complete mode directory.


def test_parallel_column_pixel_cap_is_not_retried_as_memory_shortage(tmp_path, monkeypatch):
    import temsim.physics.column_wave as column
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.wave_grid import WaveGridNumerics, WaveGridBudgetError, WaveMemoryBudgetError
    source, prepared = _refining_column_modes()
    store = ExecutedWaveStore(tmp_path, "hard-grid-cap", 1<<28)
    writers, original_writer = [], store.writer
    def track_writer(*args):
        writer = original_writer(*args)
        writers.append(writer)
        return writer
    monkeypatch.setattr(store, "writer", track_writer)
    monkeypatch.setattr(column, "numerical_thread_budget", lambda requested=None: 2)
    observed = _track_real_column_executor(monkeypatch)
    with pytest.raises(ValueError, match="Wave refinement budget exceeded") as failure:
        column._propagate_column_segmented(default_state(), source, 2000.1, store=store,
            segment_steps=1, _prepared=prepared,
            grid_numerics=WaveGridNumerics(maximum_pixels=64, maximum_working_bytes=128*1024**2))
    error = failure.value
    while error.__cause__ is not None:
        error = error.__cause__
    assert isinstance(error, WaveGridBudgetError) and not isinstance(error, WaveMemoryBudgetError)
    assert len(writers) == 1 and not writers[0]._published
    assert observed["workers"] == [2] and observed["active"] == 0
    assert not list(tmp_path.iterdir())
