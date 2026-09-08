"""Registered atom/probe tests, independent of detector names or display LUTs."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics import stem_wave_imaging as stem
from temsim.specimen.presets import load_specimen_preset


def incident_probe():
    return SimpleNamespace(
        alive=np.ones(5, dtype=bool), ray_weight=np.ones(5) / 5,
        x=np.zeros((1, 5)), y=np.zeros((1, 5)),
        tx=np.array([[0, .025, -.025, 0, 0]]),
        ty=np.array([[0, 0, 0, .025, -.025]]),
    )


def test_registered_weak_atom_has_bright_adf_and_dark_bf(monkeypatch):
    """A weak isolated phase object, full BF disk and separated ADF band.

    This is a limiting case, not a universal assertion about thick crystals.
    Potential and probe share the documented centred real-space axes.
    """
    state = default_state()
    state.simulation_mode = "ideal"
    state.acceleration_enabled = False
    state.illumination_mode = "STEM"
    state.sample.specimen_mode = "reference"
    state.sample.specimen_preset_key = "si_110"
    state.sample.wave_multislice_enabled = False
    state.sample.wave_grid_pixels = 128
    state.sample.wave_field_of_view_angstrom = 8.0
    axis = (np.arange(128) - 64) * (8 / 128)
    x, y = np.meshgrid(axis, axis)
    potential = 100 * np.exp(-(x*x + y*y) / (2 * .15**2))
    prepared = SimpleNamespace(
        x_angstrom=axis, y_angstrom=axis,
        mean_projected_potential_v_angstrom=potential,
        potential_configurations_v_angstrom=(potential,),
        slice_thicknesses_angstrom=None,
        metrics={"atomistic_applied": False, "frozen_phonon_applied": False,
                 "calculation_roi_centre_nm": (0., 0.)},
    )
    monkeypatch.setattr(stem, "_wave_grid", lambda *a: (load_specimen_preset("si_110"), prepared))
    result = stem.simulate_angle_resolved_stem(
        state, SimpleNamespace(incident=incident_probe()),
        (stem.AngularDetector("df", 35., 90.), stem.AngularDetector("bf", 0., 25.)),
        np.array([[0., .0002, -.0002]]), np.zeros((1, 3)),
    )
    bf, df = result.fractions["bf"][0], result.fractions["df"][0]
    assert df[0] > 5 * max(df[1:])
    assert bf[0] < min(bf[1:])
    np.testing.assert_allclose(df[1], df[2], rtol=1e-8, atol=1e-12)
    assert result.metrics["angular_coverage_complete"]

    # Independent abTEM probe construction and detector integration, with
    # explicitly registered box coordinates. A 0.01 A single slice approaches
    # the thin-phase limit used above. Tolerances include angle-grid rounding
    # and the reference backend's float32 transmission function.
    abtem = pytest.importorskip("abtem")
    abtem.config.set({"diagnostics.progress_bar": False})
    reference_probe = abtem.Probe(
        energy=300_000., semiangle_cutoff=result.metrics["probe_aperture_semiangle_mrad"],
        extent=8., gpts=128, soft=False, device="cpu",
    )
    reference_potential = abtem.PotentialArray(
        potential.T[None].astype(np.float32), slice_thickness=.01, extent=(8., 8.))
    scan = abtem.CustomScan([(4., 4.), (6., 4.), (2., 4.)])
    reference = reference_probe.scan(
        reference_potential, scan=scan,
        detectors=[abtem.AnnularDetector(inner=35., outer=90.),
                   abtem.AnnularDetector(inner=0., outer=25.)],
        lazy=False,
    )
    for actual, expected in zip((df, bf), reference):
        np.testing.assert_allclose(actual, np.asarray(expected.array).ravel(), rtol=5e-3, atol=2e-5)


@pytest.mark.parametrize("periodic", [False, True])
def test_abtem_atom_potential_matches_centred_coordinate_axes(periodic):
    """An independent ASE atom position, not a peak selected from our output."""
    from temsim.specimen.atomistic import _build_one_potential
    pytest.importorskip("abtem")
    from ase import Atoms

    position = (0., 0., .5) if periodic else (3., 2., .5)
    atoms = Atoms("Si", positions=[position], cell=(8., 8., 1.), pbc=periodic)
    potential, _ = _build_one_potential(atoms, gpts_xy=(64, 64), target_slice_thickness_angstrom=1.)
    iy, ix = np.unravel_index(np.argmax(potential.sum(axis=0)), (64, 64))
    peak_xy = (np.array((ix, iy)) - 32) * .125
    expected = (0., 0.) if periodic else (-1., -2.)
    np.testing.assert_allclose(peak_xy, expected, atol=.125)


def test_finite_box_odd_grid_request_keeps_origin_on_a_pixel(monkeypatch):
    from temsim.specimen import atomistic
    pytest.importorskip("abtem")
    from ase import Atoms

    atoms = Atoms("Si", positions=[(4., 4., .5)], cell=(8., 8., 1.), pbc=False)
    monkeypatch.setattr(atomistic, "build_equilibrium_atoms", lambda *a, **k: (atoms, (8., 8., 1.)))
    result = atomistic.build_atomistic_potential_ensemble(
        load_specimen_preset("si_110"), thickness_angstrom=1.,
        field_of_view_angstrom=8., pixels=65, target_slice_thickness_angstrom=1.,
        frozen_phonon_enabled=False, frozen_phonon_configurations=1,
        thermal_sigma_override_angstrom=0., thermal_seed=0,
    )
    assert result.grid_shape_yx == (66, 66)
    peak = np.unravel_index(np.argmax(result.mean_projected_potential_v_angstrom), (66, 66))
    assert peak == (33, 33)
