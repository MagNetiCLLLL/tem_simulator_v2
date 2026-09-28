"""Small analytic/CIF fixtures; no source solve, wave solve or full column."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.specimen import projected_scattering as model
from temsim.specimen.elastic_transport import (
    screened_rutherford_angle_cdf,
    screened_rutherford_total_cross_section_cm2,
)
from temsim.specimen.geometry import quaternion_from_euler_xyz_deg


def _cif(tmp_path, *, rows="Si1 Si 0 0 0 1", a=20, group=1, gamma=90):
    path = tmp_path / "projected.cif"
    path.write_text(f"""data_projection
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {a}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma {gamma}
_symmetry_Int_Tables_number {group}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
{rows}
""", encoding="utf-8")
    return path


def _state(path, **kwargs):
    sample = SimpleNamespace(cif_path=str(path), specimen_mode="atomic", inserted=True,
                             size_x_nm=1.5, size_y_nm=1.5, thickness_nm=1.5,
                             centre_x_nm=0.0, centre_y_nm=0.0, z_mm=100.0,
                             envelope_shape="rectangle",
                             specimen_orientation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))
    vars(sample).update(kwargs)
    return SimpleNamespace(sample=sample, beam_voltage_kv=300.0)


def _simulation(sigma=0.05, *, covariance=None, centre=(0., 0.), energy=(300_000.,)*4,
                slopes=(0., 0.)):
    covariance = np.eye(2)*sigma**2 if covariance is None else np.asarray(covariance)
    shape = math.sqrt(2)*np.array(((1., 0.), (-1., 0.), (0., 1.), (0., -1.)))
    positions = shape @ np.linalg.cholesky(covariance).T + centre
    return SimpleNamespace(incident=SimpleNamespace(
        x=positions[None, :, 0]*1e-9, y=positions[None, :, 1]*1e-9,
        tx=np.full((1, 4), slopes[0]), ty=np.full((1, 4), slopes[1]),
        alive=np.ones(4, dtype=bool), ray_weight=np.full(4, .25),
        energy_offset_ev=np.asarray(energy)-300_000., kinetic_energy_ev=np.asarray(energy)[None, :],
    ))


def _evaluate(state, simulation, x_nm, y_nm=None):
    x = np.asarray(x_nm, dtype=float)
    y = np.zeros_like(x) if y_nm is None else np.asarray(y_nm, dtype=float)
    return model.projected_optical_depth(state, simulation, x*1e-3, y*1e-3)


def test_single_atom_matches_normalized_gaussian_cross_section_and_conserves(tmp_path):
    state = _state(_cif(tmp_path))
    result = _evaluate(state, _simulation(.05), np.array([[0., .05, .1]]))
    sigma_nm2 = screened_rutherford_total_cross_section_cm2(14, 300_000)*1e14
    expected = sigma_nm2/(2*math.pi*.05**2)*np.exp(-.5*(np.array([[0., .05, .1]])/.05)**2)
    np.testing.assert_allclose(result.optical_depth, expected, rtol=3e-14)
    np.testing.assert_allclose(result.direct_fraction, np.exp(-expected), rtol=3e-14)
    np.testing.assert_allclose(result.direct_fraction+result.scattered_fraction, 1., atol=1e-15)
    np.testing.assert_allclose(sum(item.scattered_fraction for item in result.elements), result.scattered_fraction)
    np.testing.assert_allclose(result.plural_event_fraction, 1-np.exp(-expected)*(1+expected), atol=2e-16)
    assert result.metrics["checkpoint"] is False
    with pytest.raises(ValueError):
        result.optical_depth.setflags(write=True)


def test_width_is_actual_executed_covariance_and_broadening_removes_atomic_contrast(tmp_path):
    state = _state(_cif(tmp_path, a=5), size_x_nm=5., size_y_nm=5., thickness_nm=1.5)
    scan = np.linspace(-.25, .25, 33)
    narrow = _evaluate(state, _simulation(.04), scan)
    broad = _evaluate(state, _simulation(.4), scan)
    narrow_contrast = np.ptp(narrow.optical_depth)/np.mean(narrow.optical_depth)
    broad_contrast = np.ptp(broad.optical_depth)/np.mean(broad.optical_depth)
    assert narrow_contrast > 4
    assert broad_contrast < .02
    np.testing.assert_allclose(broad.probe_covariance_nm2, np.eye(2)*.4**2, atol=1e-15)


def test_full_covariance_orientation_changes_probe_elongation(tmp_path):
    state = _state(_cif(tmp_path))
    covariance = np.array(((.015, .011), (.011, .015)))
    result = _evaluate(state, _simulation(covariance=covariance), [0., .1, .1], [0., .1, -.1])
    assert result.optical_depth[1] > result.optical_depth[2]
    np.testing.assert_allclose(result.probe_covariance_nm2, covariance, rtol=2e-15)


def test_no_fitted_width_for_a_degenerate_executed_probe(tmp_path):
    simulation = _simulation()
    simulation.incident.x[:] = 0
    simulation.incident.y[:] = 0
    with pytest.raises(ValueError, match="No artificial probe width"):
        _evaluate(_state(_cif(tmp_path)), simulation, [0.])


def test_symmetry_expansion_and_mixed_partial_occupancy(tmp_path):
    path = _cif(tmp_path, rows="Cu1 Cu 0 0 0 0.25\nZn1 Zn 0 0 0 0.5", a=10, group=225)
    state = _state(path, size_x_nm=1., size_y_nm=1., thickness_nm=1.)
    result = _evaluate(state, _simulation(.2), [0.])
    assert [element.atomic_number for element in result.elements] == [29, 30]
    assert result.metrics["projected_atom_species_records"] == 8
    first, second = result.elements
    # Same symmetry-expanded positions; density differs only by occupancy.
    np.testing.assert_allclose(first.optical_depth/first.mean_cross_section_nm2,
                               .5*second.optical_depth/second.mean_cross_section_nm2, rtol=1e-14)


@pytest.mark.parametrize("occupancy", [-.1, 1.1])
def test_invalid_occupancy_rejected(tmp_path, occupancy):
    state = _state(_cif(tmp_path, rows=f"Si1 Si 0 0 0 {occupancy}"))
    with pytest.raises(ValueError, match="occupanc"):
        _evaluate(state, _simulation(), [0.])


def test_finite_thickness_counts_layers_and_retraction_is_vacuum(tmp_path):
    state = _state(_cif(tmp_path, a=10), size_x_nm=.8, size_y_nm=.8, thickness_nm=1.)
    thin = _evaluate(state, _simulation(), [0.])
    state.sample.thickness_nm = 3.
    thick = _evaluate(state, _simulation(), [0.])
    np.testing.assert_allclose(thick.optical_depth, 3*thin.optical_depth, rtol=1e-14)
    assert thick.metrics["maximum_plural_event_fraction"] > thin.metrics["maximum_plural_event_fraction"]
    state.sample.inserted = False
    state.sample.cif_path = "missing.cif"
    vacuum = _evaluate(state, _simulation(), [0.])
    np.testing.assert_array_equal(vacuum.direct_fraction, [1.])
    assert vacuum.elements == ()


def test_rotating_actual_sites_changes_columns_without_deforming_cell(tmp_path):
    state = _state(_cif(tmp_path, rows="Si1 Si 0 0 0 1\nSi2 Si .25 0 0 1"))
    original = _evaluate(state, _simulation(.05), [0.])
    state.sample.specimen_orientation_quaternion_wxyz = quaternion_from_euler_xyz_deg((0., 90., 0.))
    rotated = _evaluate(state, _simulation(.05), [0.])
    np.testing.assert_allclose(rotated.optical_depth, 2*original.optical_depth, rtol=1e-13)
    assert original.identity != rotated.identity


def test_absolute_scan_centres_translate_with_specimen_without_centroid_added_twice(tmp_path):
    state = _state(_cif(tmp_path))
    before = _evaluate(state, _simulation(centre=(10., -8.)), [0., .1], [0., .02])
    state.sample.centre_x_nm, state.sample.centre_y_nm = 30., -20.
    after = _evaluate(state, _simulation(centre=(10., -8.)), [30., 30.1], [-20., -19.98])
    np.testing.assert_allclose(before.optical_depth, after.optical_depth, rtol=1e-12)
    assert after.probe_centroid_nm == pytest.approx((10., -8.))


def test_disk_crops_actual_corner_sites(tmp_path):
    state = _state(_cif(tmp_path, rows="Si1 Si 0 0 0 1\nSi2 Si .3 .3 0 1"))
    rectangle = _evaluate(state, _simulation(.05), [.6], [.6])
    state.sample.envelope_shape = "disk"
    disk = _evaluate(state, _simulation(.05), [.6], [.6])
    expected_peak = screened_rutherford_total_cross_section_cm2(14, 300_000)*1e14/(2*math.pi*.05**2)
    assert rectangle.optical_depth[0] == pytest.approx(expected_peak, rel=1e-12)
    assert disk.optical_depth[0] < 1e-30


def test_original_oblique_cell_matches_independent_small_enumeration(tmp_path):
    from ase.io import read
    state = _state(_cif(tmp_path, a=10, gamma=60), size_x_nm=1.8, size_y_nm=1.8, thickness_nm=.8)
    points = np.array(([-.3, .1], [.2, .25], [.5, -.4]))
    result = _evaluate(state, _simulation(.15), points[:, 0], points[:, 1])
    cell = read(state.sample.cif_path).cell.array*.1
    integer = np.array([(i, j, k) for i in range(-4, 5) for j in range(-4, 5) for k in range(-4, 5)])
    positions = integer @ cell
    half = np.array((.9, .9, .4))
    positions = positions[np.all((positions >= -half) & (positions < half), axis=1)]
    delta = points[:, None, :] - positions[None, :, :2]
    density = np.exp(-np.sum(delta**2, axis=-1)/(2*.15**2)).sum(axis=1)/(2*math.pi*.15**2)
    expected = density*screened_rutherford_total_cross_section_cm2(14, 300_000)*1e14
    np.testing.assert_allclose(result.optical_depth, expected, rtol=5e-14)


def test_actual_incident_energy_mixture_not_nominal_ht(tmp_path):
    state = _state(_cif(tmp_path))
    state.beam_voltage_kv = 100.
    energies = (200_000., 200_000., 300_000., 300_000.)
    result = _evaluate(state, _simulation(energy=energies), [0.])
    expected = np.mean([screened_rutherford_total_cross_section_cm2(14, e)*1e14 for e in energies])
    assert result.elements[0].mean_cross_section_nm2 == pytest.approx(expected, rel=1e-14)
    assert result.mean_energy_ev == 250_000.
    assert result.metrics["energy_range_ev"] == (200_000., 300_000.)


def test_cache_uses_cif_bytes_geometry_probe_and_scan(tmp_path):
    state = _state(_cif(tmp_path))
    first = _evaluate(state, _simulation(), [0.])
    assert _evaluate(state, _simulation(), [0.]) is first
    assert _evaluate(state, _simulation(.06), [0.]).identity != first.identity
    assert _evaluate(state, _simulation(), [.1]).identity != first.identity
    _cif(tmp_path, rows="Si1 Si 0 0 0 0.5")
    changed = _evaluate(state, _simulation(), [0.])
    assert changed.identity != first.identity
    np.testing.assert_allclose(changed.optical_depth, first.optical_depth*.5, rtol=1e-14)


def test_scan_grid_order_and_size_do_not_change_point_density(tmp_path):
    state = _state(_cif(tmp_path, a=5), thickness_nm=1.)
    simulation = _simulation(.1)
    single = _evaluate(state, simulation, [.07], [.12])
    multiple = _evaluate(state, simulation, [[2., .07], [-1., .07]], [[3., .12], [-3., .12]])
    np.testing.assert_allclose(multiple.optical_depth[[0, 1], [1, 1]],
                               np.repeat(single.optical_depth, 2), rtol=3e-14)


def test_angular_quadrature_integrates_existing_cdf_and_retains_backward_probability(tmp_path):
    result = _evaluate(_state(_cif(tmp_path)), _simulation(energy=(200_000.,)*2+(300_000.,)*2), [0.])
    element = result.elements[0]
    edges = np.array([0., .01, .02, .1, math.pi/2, math.pi])
    theta, phi, weights = model.screened_angular_quadrature(element, edges, azimuth_samples=12)
    mass = weights.reshape(-1, 12).sum(axis=1)
    cdf = sum(weight*np.array([screened_rutherford_angle_cdf(edge, 14, energy) for edge in edges])
              for energy, weight in zip(element.energy_ev, element.cross_section_weights, strict=True))
    # np.sin and math.sin differ by an ULP; CDF subtraction near one has an
    # absolute float64 error, not relative accuracy in a tiny backscatter bin.
    np.testing.assert_allclose(mass, np.diff(cdf), rtol=2e-14, atol=4*np.finfo(float).eps)
    assert weights.sum() == pytest.approx(1., abs=3e-16)
    assert weights[theta > math.pi/2].sum() > 0
    assert np.all((theta.reshape(-1, 12)[:, 0] > edges[:-1]) & (theta.reshape(-1, 12)[:, 0] < edges[1:]))
    assert 0 < phi.min() < phi.max() < 2*math.pi


def test_budget_is_explicit_and_does_not_silently_replace_atoms(tmp_path, monkeypatch):
    state = _state(_cif(tmp_path), size_x_nm=11.)
    monkeypatch.setattr(model, "MAX_CANDIDATE_SITES", 1)
    with pytest.raises(ValueError, match="No bulk or sharpened-probe substitute"):
        _evaluate(state, _simulation(.12), [0.])


def test_correlated_gaussian_rectangle_overlap_matches_independent_2d_quadrature(tmp_path):
    sample = _state(_cif(tmp_path), size_x_nm=.8, size_y_nm=.6).sample
    covariance = np.array(((.08, .04), (.04, .05)))
    point = np.array([[.22, -.15]])
    actual = model._material_overlap(sample, point, covariance, (0., 0.))[0]
    nodes, weights = np.polynomial.legendre.leggauss(100)
    xx, yy = np.meshgrid(nodes*.4, nodes*.3)
    delta = np.stack((xx-point[0, 0], yy-point[0, 1]), axis=-1)
    density = np.exp(-.5*np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance), delta))
    density /= 2*math.pi*math.sqrt(np.linalg.det(covariance))
    expected = np.sum(density*weights[:, None]*weights[None, :])*.4*.3
    assert actual == pytest.approx(expected, abs=3e-11)


def test_correlated_gaussian_disk_overlap_matches_independent_polar_quadrature(tmp_path):
    sample = _state(_cif(tmp_path), size_x_nm=.8, size_y_nm=.6, envelope_shape="disk").sample
    covariance = np.array(((.08, .04), (.04, .05)))
    point = np.array([[.22, -.15]])
    actual = model._material_overlap(sample, point, covariance, (0., 0.))[0]
    nodes, weights = np.polynomial.legendre.leggauss(120)
    radius = .5*(nodes+1)
    angle = np.arange(256)*2*math.pi/256
    xx, yy = .4*radius[:, None]*np.cos(angle), .3*radius[:, None]*np.sin(angle)
    delta = np.stack((xx-point[0, 0], yy-point[0, 1]), axis=-1)
    density = np.exp(-.5*np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance), delta))
    density /= 2*math.pi*math.sqrt(np.linalg.det(covariance))
    expected = np.sum(density*radius[:, None]*weights[:, None])*.4*.3*.5*2*math.pi/256
    assert actual == pytest.approx(expected, abs=3e-11)


def test_tilted_rectangular_slab_overlap_includes_projected_thickness(tmp_path):
    from scipy.special import ndtr
    sample = _state(_cif(tmp_path), size_x_nm=.8, size_y_nm=.6, thickness_nm=2.).sample
    covariance = np.diag((.04, .09))
    point = np.array([[.2, .4]])
    actual = model._material_overlap(sample, point, covariance, (0., .2))[0]
    expected = (ndtr((.4-.2)/.2)-ndtr((-.4-.2)/.2))*(ndtr((.5-.4)/.3)-ndtr((-.5-.4)/.3))
    assert actual == pytest.approx(expected, abs=3e-11)


def test_finite_overlap_partitions_vacuum_material_and_scattered_current(tmp_path):
    state = _state(_cif(tmp_path), size_x_nm=.6, size_y_nm=.6)
    result = _evaluate(state, _simulation(.25), [0., .3, .6, 1.0])
    assert np.all(result.scattered_fraction <= result.material_overlap)
    assert np.all(result.plural_event_fraction <= result.scattered_fraction)
    np.testing.assert_allclose(result.vacuum_fraction+result.direct_material_fraction+result.scattered_fraction,
                               np.ones(4), atol=2e-16)
    assert result.material_overlap[0] > result.material_overlap[-1]


def test_material_support_cannot_be_silently_omitted(tmp_path):
    state = _state(_cif(tmp_path), eds_support_material_key="copper")
    with pytest.raises(ValueError, match="Material particle paths"):
        _evaluate(state, _simulation(), [0.])


def test_cancellation_during_projection_does_not_leave_request_context_or_cache(tmp_path):
    state = _state(_cif(tmp_path, a=17))
    calls = []
    def cancelled():
        calls.append(True)
        return len(calls) >= 3
    state._tuning_cancelled = cancelled
    with pytest.raises(RuntimeError, match="Superseded"):
        _evaluate(state, _simulation(.071), [0.])
    assert len(calls) == 3
    assert model._CANCEL_CHECK.get() is None
    state._tuning_cancelled = lambda: False
    result = _evaluate(state, _simulation(.071), [0.])
    assert result.optical_depth[0] > 0


def test_gaussian_sum_polls_cancellation_between_bounded_blocks():
    def cancelled():
        raise RuntimeError("cancel Gaussian")
    token = model._CANCEL_CHECK.set(cancelled)
    try:
        with pytest.raises(RuntimeError, match="cancel Gaussian"):
            model._gaussian_density(np.zeros((1, 2)), np.ones(1), np.zeros((2, 2)), np.eye(2))
    finally:
        model._CANCEL_CHECK.reset(token)


def test_tilted_uniform_bulk_has_one_path_length_factor_not_zero_or_two(tmp_path):
    # A probe much wider than the periodic cell samples uniform projected
    # occupancy; every z plane translates under shear without changing its
    # mean density per laboratory XY area. Both traces contain the same layers.
    state = _state(_cif(tmp_path, a=2), size_x_nm=8., size_y_nm=8., thickness_nm=2.02)
    normal = _evaluate(state, _simulation(.4), [0.])
    slopes = (.5, -.3)
    tilted = _evaluate(state, _simulation(.4, slopes=slopes), [0.])
    factor = math.sqrt(1+slopes[0]**2+slopes[1]**2)
    np.testing.assert_allclose(tilted.optical_depth/normal.optical_depth, [factor], rtol=1e-11)


def test_angular_quadrature_retains_positive_ulp_bin_and_normalized_energy_mixture():
    # Geometric detector hints can sit one float64 value from a log-grid
    # angle. The bin is tiny but its elastic probability remains positive.
    element = model.ProjectedElement(14, np.zeros(1), np.zeros(1), 1.,
                                     np.array([100_000., 300_000.]), np.array([.3, .7]))
    edges = np.array([0., .1, np.nextafter(.1, math.inf), math.pi])
    theta, _, weights = model.screened_angular_quadrature(element, edges, azimuth_samples=7)
    masses = weights.reshape(-1, 7).sum(axis=1)
    u = np.sin(edges*.5)**2
    delta = np.array([model.screened_rutherford_parameter(14, energy)
                      for energy in element.energy_ev])
    expected = np.sum(element.cross_section_weights*(1+delta)*delta*(u[2]-u[1])
                      / ((u[2]+delta)*(u[1]+delta)))
    assert masses[1] > 0
    assert masses[1] == pytest.approx(expected, rel=3e-15, abs=0.)
    assert masses.sum() == pytest.approx(1., abs=5e-16)
    assert np.all(weights >= 0)
    assert np.all(theta[7:14] >= edges[1]-np.spacing(edges[1]))
    assert np.all(theta[7:14] <= edges[2]+np.spacing(edges[2]))
