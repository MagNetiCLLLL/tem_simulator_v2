"""Display provenance resolution is pure and never invents a calculated ROI."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.sample_display_source import resolve_sample_display_source
from temsim.optics.model import Sample


def _state():
    return SimpleNamespace(sample=Sample())


def _product(bounds=(-2.0, 3.0, -4.0, 5.0), **overrides):
    fields = dict(
        metrics={
            "specimen_wave_window_bounds_nm": bounds,
            "specimen_atom_generation_bounds_nm": (-1.0, 1.0, -1.0, 1.0),
        },
        scan_x_um=np.asarray([[0.1, 0.2]]),
        scan_y_um=np.asarray([[0.3, 0.4]]),
        probe_state=SimpleNamespace(centroid_nm=(3.0, 4.0), radius_99_nm=9.0),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _result(state, *, stem=None, tem=None):
    return SimpleNamespace(
        state_snapshot=deepcopy(state), stem_scan=stem, wave_imaging=tem,
    )


def test_completed_stem_uses_captured_wave_window_not_geometric_padding():
    state = _state()
    stem = _product()
    result = _result(state, stem=stem)
    source = resolve_sample_display_source(state, result, result_is_current=True)

    assert source.sample is result.state_snapshot.sample
    assert source.calculation_roi_bounds_nm_override == (-2.0, 3.0, -4.0, 5.0)
    assert source.atom_generation_bounds_nm == (-1.0, 1.0, -1.0, 1.0)
    assert source.current_probe_nm == (3.0, 4.0)
    assert source.probe_padding_nm == 27.0
    assert source.completed_region and source.product_kind == "STEM"
    assert source.provenance_label == "Current STEM calculation region"
    assert not source.draft_sample_differs
    assert np.shares_memory(source.scan_x_um, stem.scan_x_um)
    assert not source.scan_x_um.flags.writeable
    assert stem.scan_x_um.flags.writeable  # Resolution did not mutate input flags.
    with pytest.raises(FrozenInstanceError):
        source.completed_region = False


def test_tem_completed_region_does_not_borrow_unrelated_preview_scan():
    state = _state()
    result = _result(state, stem=_product(metrics={}), tem=_product())
    source = resolve_sample_display_source(state, result)

    assert source.product_kind == "TEM"
    assert source.provenance_label == "Last completed TEM calculation region"
    assert source.scan_x_um is source.scan_y_um is source.current_probe_nm is None
    assert source.probe_padding_nm == 0.0


def test_both_complete_products_choose_stem_explicitly():
    state = _state()
    source = resolve_sample_display_source(
        state, _result(state, stem=_product(), tem=_product((-10., 10., -10., 10.))),
    )
    assert source.product_kind == "STEM"
    assert source.calculation_roi_bounds_nm_override == (-2., 3., -4., 5.)
    assert "TEM also has a completed region" in source.provenance_detail


def test_draft_sample_edits_cannot_relabel_retained_result_current():
    state = _state()
    result = _result(state, tem=_product())
    state.sample.thickness_nm = 80.0
    source = resolve_sample_display_source(state, result, result_is_current=True)

    assert source.sample.thickness_nm == result.state_snapshot.sample.thickness_nm
    assert source.draft_sample_differs
    assert source.provenance_label == "Previous TEM calculation region"
    assert "draft sample differs" in source.provenance_detail


def test_lens_only_stale_status_retains_region_without_claiming_current():
    state = _state()
    source = resolve_sample_display_source(
        state, _result(state, stem=_product()), result_is_current=False,
    )
    assert not source.draft_sample_differs
    assert source.provenance_label == "Previous STEM calculation region"
    assert source.completed_region


@pytest.mark.parametrize("bounds", [None, (), (0., 1.), (0., 0., 0., 1.),
                                   (0., 1., 0., np.nan), "invalid"])
def test_legacy_or_invalid_metadata_is_only_an_estimated_scan_region(bounds):
    state = _state()
    source = resolve_sample_display_source(state, _result(state, stem=_product(bounds)))
    assert not source.completed_region
    assert source.calculation_roi_bounds_nm_override is None
    assert source.provenance_label.startswith("Estimated scan region")
    assert "not evidence" in source.provenance_detail


def test_metadata_without_captured_sample_is_not_authoritative():
    state = _state()
    result = SimpleNamespace(wave_imaging=_product(), stem_scan=None)
    source = resolve_sample_display_source(state, result)
    assert source.sample is state.sample
    assert source.provenance_label == "Structural preview"
    assert not source.completed_region


def test_vacuum_window_keeps_actual_bounds_and_does_not_invent_material():
    state = _state()
    state.sample.specimen_preset_key = "vacuum"
    tem = _product((10., 20., 30., 40.))
    tem.metrics["specimen_atom_generation_bounds_nm"] = None
    source = resolve_sample_display_source(state, _result(state, tem=tem))
    assert source.calculation_roi_bounds_nm_override == (10., 20., 30., 40.)
    assert source.atom_generation_bounds_nm is None
    assert source.completed_region


def test_missing_external_cif_is_not_read_by_source_resolution():
    state = _state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "missing-captured-input.cif"
    source = resolve_sample_display_source(state, _result(state, tem=_product()))
    assert source.completed_region
    assert source.sample.cif_path == "missing-captured-input.cif"
    assert "reconstruction" in source.provenance_detail


def test_plain_draft_and_invalid_scan_do_not_produce_a_fictitious_region():
    state = _state()
    source = resolve_sample_display_source(state)
    assert source.provenance_label == "Structural preview"
    assert not source.completed_region
    invalid = _product(metrics={}, scan_x_um=np.array([np.nan]), scan_y_um=np.array([0.]))
    source = resolve_sample_display_source(state, _result(state, stem=invalid))
    assert source.provenance_label.startswith("Structural preview")
    assert source.scan_x_um is source.scan_y_um is None


def test_source_change_without_completed_wave_previews_current_structure():
    state = _state()
    result = _result(state, stem=_product(metrics={}))
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "newly-selected.cif"
    source = resolve_sample_display_source(state, result)
    assert source.sample is state.sample
    assert source.sample.cif_path == "newly-selected.cif"
    assert source.scan_x_um is source.scan_y_um is None
    assert source.provenance_label == "Structural preview"
    assert "current draft structure" in source.provenance_detail
    assert not source.completed_region
