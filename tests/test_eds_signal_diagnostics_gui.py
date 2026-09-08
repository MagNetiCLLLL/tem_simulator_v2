"""Completed EDS estimates distinguish missed material from physical zero."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.gui.eds_panel import EDSPage
from temsim.optics.column import default_state


def _sampling_metrics(*, hits=0, status="no_sampled_material_hits"):
    return {
        "positive_weight_trajectory_count": 193,
        "material_hit_trajectory_count": hits,
        "sample_hit_trajectory_count": hits,
        "material_hit_weight_fraction": .0014 if hits else 0.,
        "sample_hit_weight_fraction": .0014 if hits else 0.,
        "material_sampling_status": status,
        "incident_position_rms_radius_nm": 220.,
        "sample_size_xy_nm": (10., 10.),
        "sample_thickness_nm": 5.,
    }


def _spectrum(metrics, *, expected=(0., 0., 0.), sampled=None, elastic=None):
    values = np.asarray(expected, dtype=float)
    return SimpleNamespace(
        energy_bin_centres_ev=np.asarray((1000., 1100., 1200.)),
        expected_counts=values,
        sampled_counts=None if sampled is None else np.asarray(sampled, dtype=float),
        total_expected_counts=float(np.sum(values)),
        lines=(), metrics=dict(metrics), elastic_transport=elastic,
    )


@pytest.fixture
def page(qtbot):
    widget = EDSPage()
    qtbot.addWidget(widget)
    state = default_state()
    state.sample.eds_enabled = True
    widget.set_state(state)
    widget.resize(1000, 650)
    widget.show()
    return widget


def _publish(page, spectrum):
    snapshot = type(page._state).from_dict(page._state.to_dict())
    result = SimpleNamespace(
        specimen_interactions=None if spectrum is None else SimpleNamespace(eds_spectrum=spectrum),
        simulation=SimpleNamespace(incident=SimpleNamespace(alive=np.asarray([True]), ray_weight=np.asarray([1.]))),
        state_snapshot=snapshot,
    )
    page.display_result(result)
    return result


def test_weighted_overlap_explains_positive_eds_with_original_misses(page):
    metrics = {
        **_sampling_metrics(),
        "eds_overlap_sampling_status": "active",
        "eds_overlap_material_weight_fraction": 3.e-6,
        "eds_overlap_sampling_point_count": 256,
    }
    _publish(page, _spectrum(metrics, expected=(0., .01, 0.)))
    text = page.signal_diagnostics.text()
    assert "Weighted overlap (EDS only): 0.0003%" in text
    assert "256 integration points" in text
    assert "KDE approximation" in text
    assert "Original rays missed material" in text
    assert "zero estimate" not in text
    assert "not renormalised to full beam current" in text


def test_overlap_fallback_and_settings_are_visible(page):
    assert page.eds_overlap_sampling_enabled.isChecked()
    assert page.eds_overlap_sampling_points.value() == 256
    page.eds_overlap_sampling_enabled.setChecked(False)
    assert not page._state.sample.eds_overlap_sampling_enabled
    assert not page.eds_overlap_sampling_points.isEnabled()
    page.eds_overlap_sampling_enabled.setChecked(True)
    page.eds_overlap_sampling_points.setValue(512)
    assert page._state.sample.eds_overlap_sampling_points == 512
    _publish(page, _spectrum({
        **_sampling_metrics(),
        "eds_overlap_sampling_status": "unsupported_support",
        "eds_overlap_sampling_detail": "Material support: original ray estimate retained.",
    }))
    assert "Material support: original ray estimate retained." in page.signal_diagnostics.text()


def test_cached_zero_with_no_sampled_material_hits_is_visibly_unresolved(page, qtbot):
    spectrum = _spectrum(_sampling_metrics())
    before_metrics = deepcopy(spectrum.metrics)
    before_counts = spectrum.expected_counts.copy()
    _publish(page, spectrum)
    label = page.signal_diagnostics
    qtbot.waitUntil(lambda: label.isVisible() and label.height() >= label.heightForWidth(label.width()))
    text = label.text()
    assert "Expected: 0 counts" in text
    assert "No sampled ray crossed material; zero estimate does not establish zero physical signal." in text
    assert "Check overlap / increase ray sampling." in text
    assert "Material hits: 0/193 positive-weight rays (0% beam weight)" in text
    assert "Incident RMS radius: 220 nm" in text
    assert "Sample size: 10 \u00d7 10 \u00d7 5 nm" in text
    page._state.sample.size_x_nm = 900.
    assert label.text() == text  # Diagnostics belong to the displayed snapshot.
    assert label.wordWrap() and label.height() >= label.heightForWidth(label.width())
    assert page.width() == 1000
    np.testing.assert_array_equal(page._spectrum_counts, before_counts)
    np.testing.assert_array_equal(page.spectrum_plot.listDataItems()[0].yData, before_counts)
    assert spectrum.metrics == before_metrics


def test_manual_zero_acquisition_uses_same_diagnostics_without_calling_it_vacuum(page, monkeypatch):
    from temsim.specimen import interaction_engine

    metrics = {
        **_sampling_metrics(),
        "trajectory_count": 193,
        "mean_elastic_events_per_trajectory": 0.,
        "transmitted_fraction": 1.,
        "backscattered_fraction": 0.,
        "stored_trajectory_count": 0,
        "rutherford_heavy_element_warning": False,
    }
    elastic = SimpleNamespace(metrics=metrics)
    # The elastic result can supply diagnostics before old spectrum metrics
    # have incorporated the new fields.
    spectrum = _spectrum({}, elastic=elastic)
    interactions = SimpleNamespace(eds_spectrum=spectrum, metrics={"calculated_observables_this_call": ("characteristic_x_ray",)})
    result = _publish(page, None)
    catalog = AssemblyCatalog()
    result.assembly = catalog.apply(result.state_snapshot, catalog.default_selection())
    received = []
    monkeypatch.setattr(interaction_engine, "run_specimen_interactions", lambda state, *args, **kwargs: (received.append(state) or interactions))
    errors = []
    page.error.connect(errors.append)
    assert page.eds_acquire.isEnabled()
    page.eds_acquire.click()
    assert not errors
    assert received == [result.state_snapshot]
    assert "No sampled ray crossed material" in page.signal_diagnostics.text()
    assert "no characteristic contributions" in page.eds_summary.text()
    assert "vacuum only" not in page.eds_summary.text()
    assert "vacuum only" not in page.eds_summary.toolTip()
    np.testing.assert_array_equal(page._spectrum_counts, np.zeros(3))


@pytest.mark.parametrize("sampled", [None, (0., 0., 0.)])
def test_sparse_crossings_and_poisson_zero_are_separate_from_expected_counts(page, sampled):
    spectrum = _spectrum(_sampling_metrics(hits=2, status="sampled_material_hits"),
                         expected=(.01, .02, .03), sampled=sampled)
    _publish(page, spectrum)
    text = page.signal_diagnostics.text()
    assert "Expected: 0.06 counts" in text
    assert "Material hits: 2/193 positive-weight rays (0.14% beam weight)" in text
    assert "Sample hits: 2/193 positive-weight rays (0.14% beam weight)" in text
    assert "No sampled ray crossed material" not in text
    if sampled is None:
        assert "Poisson sampled" not in text
        assert "Poisson sample is zero" not in text
        assert page._spectrum_count_label == "Expected counts"
        np.testing.assert_array_equal(page._spectrum_counts, spectrum.expected_counts)
    else:
        assert "Poisson sampled: 0 counts" in text
        assert "Poisson sample is zero despite a nonzero expected signal." in text
        assert page._spectrum_count_label == "Sampled counts"
        np.testing.assert_array_equal(page._spectrum_counts, spectrum.sampled_counts)
    np.testing.assert_array_equal(spectrum.expected_counts, np.array([.01, .02, .03]))


def test_legacy_metrics_do_not_invent_zero_material_hits(page):
    _publish(page, _spectrum({"trajectory_count": 193}))
    text = page.signal_diagnostics.text()
    assert "Expected: 0 counts" in text
    assert "Material-hit diagnostics unavailable for this result." in text
    assert "0/193" not in text
    assert "No sampled ray crossed material" not in text
    assert "Transport reports no material" not in text
    assert "X-rays generated, none collected" not in text


@pytest.mark.parametrize("blocked", [None, {"objective_aperture": 6, "holder": 2}])
def test_generated_but_uncollected_photons_only_name_recorded_obstructions(page, blocked):
    metrics = {
        **_sampling_metrics(hits=2, status="sampled_material_hits"),
        "total_expected_emitted_photons": 12.,
        "photon_transport_expected_unattenuated_counts": .4,
        "photon_transport_expected_detected_counts": 0.,
    }
    if blocked is not None:
        metrics["photon_transport_blocked_by_component_counts"] = blocked
        metrics["photon_transport_outside_acceptance_count"] = 3
    _publish(page, _spectrum(metrics))
    text = page.signal_diagnostics.text()
    assert "X-rays generated, none collected." in text
    assert "Generated photons: 12" in text
    assert "Unattenuated expected counts: 0.4" in text
    assert "No sampled ray crossed material" not in text
    if blocked is None:
        assert "Blocked photon paths" not in text
        assert "outside acceptance" not in text
    else:
        assert "objective_aperture (6)" in text and "holder (2)" in text
        assert "Photon paths outside acceptance: 3" in text
    np.testing.assert_array_equal(page._spectrum_counts, np.zeros(3))


@pytest.mark.parametrize("collection_metrics", [
    {"photon_transport_expected_detected_counts": .4, "counts_outside_spectrum": .4},
    {"photon_transport_expected_detected_counts": .4},
    {"counts_outside_spectrum": .4},
])
def test_narrow_energy_window_does_not_mislabel_collected_xrays_as_uncollected(page, collection_metrics):
    # The displayed 1.0--1.2 keV bins exclude a collected Si K-alpha peak
    # near 1.74 keV. A zero visible spectrum is not zero photon collection.
    spectrum = _spectrum({
        **_sampling_metrics(hits=2, status="sampled_material_hits"),
        "total_expected_emitted_photons": 12.,
        **collection_metrics,
    })
    _publish(page, spectrum)
    text = page.signal_diagnostics.text()
    assert "Expected: 0 counts" in text
    assert "Collected X-rays lie outside the displayed energy range." in text
    assert "none collected" not in text
    np.testing.assert_array_equal(page._spectrum_counts, np.zeros(3))
    np.testing.assert_array_equal(spectrum.energy_bin_centres_ev, np.array([1000., 1100., 1200.]))


def test_old_generated_xray_metrics_do_not_infer_missing_collection_counts(page):
    _publish(page, _spectrum({"total_expected_emitted_photons": 12.}))
    text = page.signal_diagnostics.text()
    assert "collection diagnostics unavailable" in text
    assert "none collected" not in text
    assert "outside the displayed energy range" not in text


def test_explicit_no_material_is_distinct_from_a_finite_material_sampling_miss(page):
    _publish(page, _spectrum(_sampling_metrics(status="no_material")))
    assert "Transport reports no material." in page.signal_diagnostics.text()
    assert "zero estimate does not establish zero physical signal" not in page.signal_diagnostics.text()


def test_stale_and_cleared_spectra_never_leave_current_diagnostics(page):
    assert page.signal_diagnostics.isHidden()
    spectrum = _spectrum(_sampling_metrics())
    _publish(page, spectrum)
    original = page._spectrum_counts.copy()
    page.mark_result_stale()
    assert "Previous EDS result | inputs changed" in page.signal_diagnostics.text()
    assert "No sampled ray crossed material" in page.signal_diagnostics.text()
    np.testing.assert_array_equal(page._spectrum_counts, original)
    assert not page.eds_acquire.isEnabled()
    _publish(page, None)
    assert page.signal_diagnostics.isHidden() and page.signal_diagnostics.text() == ""
    assert page._spectrum_counts.size == 0
    _publish(page, spectrum)
    assert "Previous EDS result" not in page.signal_diagnostics.text()
    page.mark_result_stale()
    page.display_result(None)
    assert page.signal_diagnostics.isHidden() and page.signal_diagnostics.text() == ""
    assert page.spectrum_plot.listDataItems() == []
