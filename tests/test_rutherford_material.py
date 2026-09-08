"""Small crystal/material checks, no wave solve or Monte Carlo workload."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import pytest
from ase.data import atomic_masses
from scipy.constants import alpha, c, electron_mass, hbar, physical_constants

from temsim.detector.eds_signal import material_from_cif
from temsim.specimen.rutherford import (
    ATOMIC_MASS_UNIT_G, TailElement, TailMaterial, build_tail_angular_distribution,
    estimate_screening_angle_mrad, read_cif_composition, resolve_tail_material,
)
from temsim.specimen.virtual import physical_screened_rutherford_probability


def _cif(tmp_path, rows="Si1 Si 0 0 0 1", *, group=1, a=5.0):
    path = tmp_path / "material.cif"
    path.write_text(f"""data_test
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {a}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_Int_Tables_number {group}
_chemical_formula_sum 'Si8'
_cell_formula_units_Z 8
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


def _sample(path="", **overrides):
    return SimpleNamespace(specimen_mode="atomic", cif_path=str(path),
                           **overrides)


def test_symmetry_expands_sites_without_multiplying_formula_z_again(tmp_path):
    path = _cif(tmp_path, group=227, a=5.44370237)
    material = read_cif_composition(path)
    assert material.atoms_per_cell == ((14, 8.0),)
    assert material.volume_nm3 == pytest.approx(0.544370237 ** 3)
    assert dict(material.number_densities_atoms_nm3)[14] == pytest.approx(49.59145716751146)
    assert not material.partial_occupancy and not material.mixed_occupancy
    tail = resolve_tail_material(_sample(path), 5.0, 200.0)
    assert tail.elements[0].areal_density_atoms_nm2 == pytest.approx(247.9572858375573)
    assert tail.material_source == "structure" and tail.screening_source == "moliere"


def test_mixed_and_partial_occupancy_are_counted_for_every_expanded_site(tmp_path):
    path = _cif(tmp_path, "Cu1 Cu 0 0 0 0.25\nZn1 Zn 0 0 0 0.5", group=225)
    composition = read_cif_composition(path)
    # Four equivalent fcc sites; ASE's displayed majority Zn is insufficient.
    assert composition.atoms_per_cell == ((29, 1.0), (30, 2.0))
    assert composition.partial_occupancy and composition.mixed_occupancy
    mass = atomic_masses[29] + 2 * atomic_masses[30]
    assert composition.density_g_cm3 == pytest.approx(mass * ATOMIC_MASS_UNIT_G / (125e-24))
    assert dict(composition.mass_fractions)[29] == pytest.approx(atomic_masses[29] / mass)
    eds = material_from_cif(path)
    assert eds.density_g_cm3 == composition.density_g_cm3
    assert eds.mass_fractions == composition.mass_fractions
    tail = resolve_tail_material(_sample(path), 5, 200)
    assert [row.atomic_number for row in tail.elements] == [29, 30]
    assert [row.areal_density_atoms_nm2 for row in tail.elements] == pytest.approx([40, 80])
    assert any("coherent atomistic wave cannot represent" in warning for warning in tail.warnings)


@pytest.mark.parametrize("occupancy", ["-0.1", "nan", "1.2"])
def test_invalid_occupancy_cannot_silently_become_a_full_atom(tmp_path, occupancy):
    with pytest.raises(ValueError, match="occupanc"):
        read_cif_composition(_cif(tmp_path, f"Si1 Si 0 0 0 {occupancy}"))


def test_overoccupied_mixture_rejected_and_file_edits_refresh_composition(tmp_path):
    with pytest.raises(ValueError, match="sum to more than one"):
        read_cif_composition(_cif(tmp_path, "Cu1 Cu 0 0 0 0.6\nZn1 Zn 0 0 0 0.6"))
    path = _cif(tmp_path)
    first = read_cif_composition(path)
    assert read_cif_composition(path) is first
    _cif(tmp_path, a=10)
    assert read_cif_composition(path).number_densities_atoms_nm3[0][1] == pytest.approx(first.number_densities_atoms_nm3[0][1] / 8)


@pytest.mark.parametrize("z,energy", [(1, 80), (14, 200), (79, 300)])
def test_moliere_angle_reconstructs_published_dimensionless_parameter(z, energy):
    # Independent SI evaluation of eq.97, checked in the actual kernel form.
    kinetic = energy * 1e3 * physical_constants["electron volt"][0]
    rest = electron_mass * c**2
    p = math.sqrt(kinetic * (kinetic + 2 * rest)) / c
    beta = p * c / (kinetic + rest)
    a_tf = 0.5 * (3 * math.pi / 4)**(2/3) * physical_constants["Bohr radius"][0] / z**(1/3)
    expected = (hbar / (2 * p * a_tf))**2 * (1.13 + 3.76 * (alpha*z/beta)**2)
    angle = estimate_screening_angle_mrad(z, energy)
    assert math.sin(angle * 1e-3 / 2)**2 == pytest.approx(expected, rel=2e-9)


def test_manual_material_and_screening_are_independent_overrides(tmp_path):
    path = _cif(tmp_path)
    manual_material = _sample("missing.cif", real_tail_material_source="manual",
                              real_tail_atomic_number=79, real_tail_areal_density_atoms_nm2=33)
    tail = resolve_tail_material(manual_material, 5, 200)
    assert tail.composition is None and tail.elements[0].areal_density_atoms_nm2 == 33
    assert tail.elements[0].screening_angle_mrad == estimate_screening_angle_mrad(79, 200)
    assert resolve_tail_material(manual_material, 100, 200).elements == tail.elements
    auto_material = _sample(path, real_tail_screening_source="manual", real_tail_screening_angle_mrad=7)
    result = resolve_tail_material(auto_material, 5, 200)
    assert result.elements[0].screening_angle_mrad == 7
    assert result.elements[0].areal_density_atoms_nm2 == pytest.approx(40)
    with pytest.raises(ValueError, match="CIF"):
        resolve_tail_material(_sample("missing.cif"), 5, 200)
    with pytest.raises(ValueError, match="positive finite areal density"):
        resolve_tail_material(_sample(real_tail_material_source="manual"), 5, 200)


def _mixture(scale=1.0):
    return TailMaterial((TailElement(14, None, 500 * scale, 5), TailElement(79, None, 100 * scale, 12)),
                        "manual", "manual", 5, "Synthetic mixture for conservation", (), None)


@pytest.mark.parametrize("scale", [1, 1e6])
def test_mixture_uses_one_total_poisson_probability_and_cross_section_weighting(scale):
    distribution = build_tail_angular_distribution(_mixture(scale), beam_energy_kv=200,
                       minimum_angle_mrad=40, maximum_angle_mrad=250)
    tau = [item.parameters["optical_depth"] for item in distribution.components]
    expected = -math.expm1(-sum(tau))
    assert distribution.scattered_probability == pytest.approx(expected)
    assert distribution.total_probability == pytest.approx(1)
    assert distribution.direct_probability >= 0
    assert sum(item.probability for item in distribution.components) == pytest.approx(expected)
    assert distribution.components[0].probability / expected == pytest.approx(tau[0]/sum(tau))
    assert distribution.scattered_probability < sum(-math.expm1(-value) for value in tau)
    assert np.min(np.hypot(distribution.angle_x_mrad, distribution.angle_y_mrad)) == pytest.approx(40)
    assert np.max(np.hypot(distribution.angle_x_mrad, distribution.angle_y_mrad)) == pytest.approx(250)
    assert not distribution.probabilities.flags.writeable


def test_single_element_manual_matches_existing_law_and_zero_thickness_is_vacuum(tmp_path):
    material = replace(_mixture(), elements=_mixture().elements[:1])
    kwargs = dict(beam_energy_kv=200, minimum_angle_mrad=50, maximum_angle_mrad=250)
    distribution = build_tail_angular_distribution(material, **kwargs)
    old_probability, old_sigma = physical_screened_rutherford_probability(
        atomic_number=14, areal_density_atoms_nm2=500, screening_angle_mrad=5, **kwargs)
    assert distribution.scattered_probability == pytest.approx(old_probability)
    assert distribution.components[0].parameters["integrated_cross_section_m2"] == old_sigma
    zero = resolve_tail_material(_sample(_cif(tmp_path)), 0, 200)
    empty = build_tail_angular_distribution(zero, **kwargs)
    assert empty.direct_probability == 1 and empty.scattered_probability == 0


@pytest.mark.parametrize("material_source", ["structure", "manual"])
def test_zero_interacting_thickness_cannot_scatter_dormant_tail_inputs(monkeypatch, material_source):
    from temsim.detector import stem_signal

    sample = _sample(real_tail_material_source=material_source,
                     real_tail_atomic_number=14, real_tail_areal_density_atoms_nm2=1e6,
                     real_high_angle_tail_enabled=True)
    state = SimpleNamespace(sample=sample)
    monkeypatch.setattr("temsim.physics.wave_imaging.effective_sample_thickness_nm", lambda _: 0.)
    def must_not_resolve(*_args, **_kwargs):
        raise AssertionError("Vacuum must not resolve dormant tail material or optics")
    monkeypatch.setattr(stem_signal, "resolve_tail_material", must_not_resolve)
    zero = np.zeros((1, 2))
    images, lost, probability, metrics = stem_signal._real_high_angle_tail(
        None, state, [SimpleNamespace(key="haadf")], zero, zero, None, 50)
    np.testing.assert_array_equal(images["haadf"], zero)
    np.testing.assert_array_equal(lost, zero)
    np.testing.assert_array_equal(probability, zero)
    assert metrics is None


@pytest.mark.parametrize("lower,upper", [(50, 50), (60, 50), (20, 501), (float("nan"), 200)])
def test_tail_bounds_reject_overlap_or_unsupported_ranges(lower, upper):
    with pytest.raises(ValueError, match="minimum < maximum"):
        build_tail_angular_distribution(_mixture(), beam_energy_kv=200,
                     minimum_angle_mrad=lower, maximum_angle_mrad=upper)


def test_stem_tail_preserves_nonoverlap_and_sequential_probability_budget(monkeypatch, tmp_path):
    from temsim.detector import stem_signal
    sample = _sample(_cif(tmp_path, "Cu1 Cu 0 0 0 0.25\nZn1 Zn 0 0 0 0.75"),
                     real_high_angle_tail_enabled=True, real_tail_max_angle_mrad=250,
                     size_x_nm=10, size_y_nm=10, centre_x_nm=0, centre_y_nm=0)
    state = SimpleNamespace(sample=sample, beam_voltage_kv=200)
    monkeypatch.setattr("temsim.physics.wave_imaging.effective_sample_thickness_nm", lambda _: 5)
    monkeypatch.setattr(stem_signal, "probe_state_from_simulation", lambda *_: SimpleNamespace(probe_sigma_nm=0))
    monkeypatch.setattr(stem_signal, "measure_sample_current", lambda *_: SimpleNamespace(fraction=.8))
    monkeypatch.setattr(stem_signal, "finite_sample_gaussian_overlap", lambda *_args, **_kwargs: np.array([[1., .25]]))
    class Detector:
        def __init__(self, key):
            self.key = key
        def acceptance_mask(self, x, y):
            return np.ones_like(x, dtype=bool)
    images, lost, probability, metrics = stem_signal._real_high_angle_tail(
        _tail_incident(), state, [Detector("first"), Detector("second")], np.zeros((1,2)), np.zeros((1,2)), None, 50,
        _tail_plan())
    assert metrics["minimum_angle_mrad"] > 50
    assert len(metrics["elements"]) == 2 and "atomic_number" not in metrics
    assert metrics["thickness_nm"] == 5
    assert np.all(images["second"] == 0) and np.all(lost == 0)
    assert np.allclose(images["first"], probability)
    assert np.all(probability <= .8)
    assert probability[0,1] == pytest.approx(probability[0,0]/4)


def _tail_incident(chief_x_rad=0., origin_x_m=0.):
    return SimpleNamespace(incident=SimpleNamespace(
        alive=np.ones(1, dtype=bool), ray_weight=np.ones(1),
        x=np.array([[origin_x_m]]), y=np.zeros((1, 1)),
        tx=np.array([[chief_x_rad]]), ty=np.zeros((1, 1)),
    ))


def _tail_plan(aperture_radius_mm=None, *, aperture_after=False):
    from temsim.physics.first_order import TransverseTransfer
    from temsim.physics.record_plane import PlaneStop, RecordPlanePlan
    planes = [PlaneStop("first", "First", 2., "detector", "disk", outer_width_mm=4., readout_enabled=True),
              PlaneStop("second", "Second", 3., "detector", "disk", outer_width_mm=4., readout_enabled=True)]
    if aperture_radius_mm is not None:
        planes.append(PlaneStop("fixed_dpa", "Fixed DPA", 4. if aperture_after else 1.,
                                "aperture", "disk", radius_mm=aperture_radius_mm))
    planes.sort(key=lambda plane: plane.z_mm)
    transfers = tuple(TransverseTransfer(0., plane.z_mm, np.eye(2), np.eye(2)*.001,
                                        np.zeros((2, 2)), np.eye(2)) for plane in planes)
    return RecordPlanePlan(0., tuple(planes), transfers, "1"*64, "2"*64)


def test_tail_uses_wave_aperture_order_and_does_not_bypass_fixed_dpa(monkeypatch):
    from temsim.detector import stem_signal
    sample = _sample(real_tail_material_source="manual", real_tail_atomic_number=14,
                     real_tail_areal_density_atoms_nm2=250, real_tail_screening_source="manual",
                     real_high_angle_tail_enabled=True, real_tail_max_angle_mrad=250,
                     size_x_nm=10, size_y_nm=10, centre_x_nm=0, centre_y_nm=0)
    state = SimpleNamespace(sample=sample, beam_voltage_kv=200)
    monkeypatch.setattr("temsim.physics.wave_imaging.effective_sample_thickness_nm", lambda _: 5)
    monkeypatch.setattr(stem_signal, "probe_state_from_simulation", lambda *_: SimpleNamespace(probe_sigma_nm=0))
    monkeypatch.setattr(stem_signal, "measure_sample_current", lambda *_: SimpleNamespace(fraction=.8))
    detectors = [SimpleNamespace(key="first"), SimpleNamespace(key="second")]
    zero = np.zeros((1, 1))
    for after in (False, True):
        plan = _tail_plan(.003, aperture_after=after)  # Only 3 mrad fits through this physical aperture.
        images, lost, probability, metrics = stem_signal._real_high_angle_tail(
            _tail_incident(), state, detectors, zero, zero, {"first": (zero+1e9, zero)}, 50, plan)
        assert probability[0,0] > 0
        assert metrics["record_plane_plan_fingerprint"] == plan.fingerprint
        assert images["second"][0,0] == 0
        if after:
            assert images["first"][0,0] == pytest.approx(probability[0,0])
            assert lost[0,0] == 0
        else:
            assert images["first"][0,0] == 0
            assert lost[0,0] == pytest.approx(probability[0,0])
        assert (images["first"] + images["second"] + lost)[0,0] == pytest.approx(probability[0,0])


def test_tilted_tail_overlap_is_removed_without_renormalising_or_double_counting(monkeypatch):
    from temsim.detector import stem_signal
    sample = _sample(real_tail_material_source="manual", real_tail_atomic_number=14,
                     real_tail_areal_density_atoms_nm2=250, real_tail_screening_source="manual",
                     real_high_angle_tail_enabled=True, real_tail_max_angle_mrad=100,
                     size_x_nm=10, size_y_nm=10, centre_x_nm=0, centre_y_nm=0)
    state = SimpleNamespace(sample=sample, beam_voltage_kv=200)
    monkeypatch.setattr("temsim.physics.wave_imaging.effective_sample_thickness_nm", lambda _: 5)
    monkeypatch.setattr(stem_signal, "probe_state_from_simulation", lambda *_: SimpleNamespace(probe_sigma_nm=0))
    monkeypatch.setattr(stem_signal, "measure_sample_current", lambda *_: SimpleNamespace(fraction=1.))
    zero = np.zeros((1, 1))
    images, lost, tail_probability, metrics = stem_signal._real_high_angle_tail(
        _tail_incident(chief_x_rad=.03), state, [SimpleNamespace(key="first")], zero, zero, None, 50, _tail_plan())
    material = resolve_tail_material(sample, 5, 200)
    original = build_tail_angular_distribution(material, beam_energy_kv=200,
                       minimum_angle_mrad=50*(1+1e-9), maximum_angle_mrad=100)
    absolute_x = original.angle_x_mrad*1e-3 + .03
    absolute_y = original.angle_y_mrad*1e-3
    overlap = np.sin(absolute_x)**2 + np.sin(absolute_y)**2 <= np.sin(.05)**2
    expected_removed = float(np.sum(original.probabilities[overlap]))
    assert expected_removed > 0
    assert metrics["overlap_removed_probability"] == pytest.approx(expected_removed)
    assert tail_probability[0,0] == pytest.approx(original.scattered_probability-expected_removed)
    assert images["first"][0,0] == pytest.approx(tail_probability[0,0])
    assert lost[0,0] == 0
    # The caller leaves removed mass in the wave budget, rather than treating
    # it as a second scattered/lost population or boosting retained angles.
    wave_scale = 1-tail_probability[0,0]
    assert wave_scale + images["first"][0,0] + lost[0,0] == pytest.approx(1)
