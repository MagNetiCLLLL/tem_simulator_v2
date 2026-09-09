"""Presentation-only cached-plane analysis; no field or specimen solve."""
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest

from temsim.gui import beam_analysis
from temsim.gui.diagnostic_tabs import TransverseBeamView, InitialDirectionColourWheel
from temsim.gui.beam_tracking_modes import branch_interaction_style
from temsim.physics.ray_identity import source_identity, select_identity
from temsim.specimen.downstream_transport import GeometricSpecimenExit


def make_result(n=12, current=200.):
    phase = np.linspace(0., 2*np.pi, n, endpoint=False)
    x, y = np.cos(phase)*2e-6, np.sin(phase)*1e-6
    def branch(name, z):
        return SimpleNamespace(
            name=name, z=np.asarray(z, float), x=np.stack((x, x*.1)),
            y=np.stack((y, y*.1)), tx=np.tile(np.cos(phase)*.01, (2, 1)),
            ty=np.tile(np.sin(phase)*.02, (2, 1)), blocked_z=np.full(n, np.nan),
            ray_weight=np.full(n, 1/n), weight=1.,
        )
    incident=branch("incident", [0., 1.])
    post=branch("000", [1., 2.])
    post.x[0]=incident.x[-1]; post.y[0]=incident.y[-1]
    post.x[1]=0.; post.y[1]=0.  # Projector focus must not erase lineage.
    metrics={} if current is None else {"effective_source_current_pa": current}
    return SimpleNamespace(simulation=SimpleNamespace(incident=incident, branches={"000":post},
        metrics=metrics), signatures={"sample_downstream":"current"})


def add_detailed_exit(res):
    source=res.simulation.incident
    ids, angles=source_identity(source)
    children=[]
    for name, parents, weight, kind in (
        ("specimen_primary:000", [0, 2, 4], .6, "sample_region_primary"),
        ("specimen_elastic:real_plasmon", [0, 2], .4, "sample_region_elastic"),
    ):
        parents=np.asarray(parents)
        selected_ids, selected_angles=select_identity(ids, angles, parents)
        children.append(SimpleNamespace(
            name=name, z=np.array([1., 2.]),
            x=np.stack((source.x[-1, parents], np.zeros(parents.size))),
            y=np.stack((source.y[-1, parents], np.zeros(parents.size))),
            tx=source.tx[:, parents], ty=source.ty[:, parents],
            blocked_z=np.full(parents.size, np.nan), weight=weight,
            ray_weight=np.full(parents.size, 1/parents.size),
            source_ray_id=selected_ids, source_azimuth_rad=selected_angles,
            interaction_kind=kind,
        ))
    res.specimen_exit=GeometricSpecimenExit(tuple(children),
        {"tracked_downstream_source_probability":1., "inelastic_absorbed_source_probability":0.},
        "current")
    return children


def switch(view, key):
    view.analysis_mode.setCurrentIndex(view.analysis_mode.findData(key))


def interaction_colours(view):
    view.colour_mode.setCurrentIndex(view.colour_mode.findData("interaction"))


@pytest.fixture
def view(qtbot):
    widget=TransverseBeamView(); qtbot.addWidget(widget)
    widget.resize(430, 900); widget.show()
    return widget


@pytest.mark.parametrize("mode", [key for _, key in beam_analysis.BeamAnalysisControls.MODES])
def test_each_function_reads_existing_result_without_mutation(view, mode):
    res=make_result(); original=res.simulation.incident.x.copy()
    view.display_result(res); switch(view, mode)
    assert "unavailable" not in view.summary.text().lower()
    assert view._result is res
    np.testing.assert_array_equal(res.simulation.incident.x, original)
    if mode == "interactions":
        assert view.analysis.table.rowCount() == 1
        assert view.analysis.table.item(0, 1).text() == "100"
        assert view.analysis.table.item(0, 2).text() == "200"
        assert view.plot.isHidden()
    elif mode == "intensity":
        assert view.analysis._hover_payload[-1] == "pA / bin"


def test_angular_projection_phase_space_and_source_colours_agree(view):
    res=make_result(); view.display_result(res)
    colour_by_id={row["source_ray_id"]: brush.color().rgba()
        for row, brush in zip(view._scatter.data["data"], view._scatter.data["brush"])}
    view.set_projection_angle(90.); switch(view, "angular")
    np.testing.assert_allclose(view._scatter.data["x"], np.arctan(res.simulation.incident.ty[-1])*1e3,
        atol=1e-12)
    np.testing.assert_allclose(view._scatter.data["y"], np.arctan(-res.simulation.incident.tx[-1])*1e3,
        atol=1e-12)
    assert view.plot.getAxis("bottom").labelUnits == "mrad"
    assert [brush.color().rgba() for brush in view._scatter.data["brush"]] == list(colour_by_id.values())
    switch(view, "phase_u")
    np.testing.assert_allclose(view._scatter.data["x"], res.simulation.incident.y[-1]*1e6, atol=1e-12)
    np.testing.assert_allclose(view._scatter.data["y"], np.arctan(res.simulation.incident.ty[-1])*1e3, atol=1e-12)
    switch(view, "phase_v")
    np.testing.assert_allclose(view._scatter.data["x"], -res.simulation.incident.x[-1]*1e6, atol=1e-12)


def test_interactions_persist_through_projector_focus_and_match_table(view):
    res=make_result(); children=add_detailed_exit(res)
    view.display_result(res); view.focus_z(2.); interaction_colours(view)
    assert list(view._scatter.data["symbol"]) == ["o"]*3 + ["star"]*2
    np.testing.assert_array_equal(view._scatter.data["x"], np.zeros(5))
    expected=branch_interaction_style(children[1])[2]
    assert view._scatter.data["brush"][-1].color().getRgb()[:3] == expected
    assert "Elastic + plasmon" in view.analysis.legend.text()
    switch(view, "angular")
    assert "np.uint8" not in view.analysis.legend.text()
    assert list(view._scatter.data["symbol"]) == ["o"]*3 + ["star"]*2
    switch(view, "interactions")
    table=view.analysis.table
    assert [table.item(i, 1).text() for i in range(2)] == ["60", "40"]
    assert [table.item(i, 2).text() for i in range(2)] == ["120", "80"]
    assert "not collision counts" in view.analysis.legend.text()


@pytest.mark.parametrize("mode", ["angular", "phase_u", "phase_v", "intensity", "angle_histogram"])
def test_z_changes_keep_per_mode_ranges_and_fit_is_explicit(view, mode):
    view.display_result(make_result())
    view._apply_centered_view_ranges(3., 3.); view._view_scale_initialized=True
    original=np.asarray(view.plot.viewRange())
    switch(view, mode)
    bounds=((-80., 80.), (-100., 100.)) if mode != "angle_histogram" else ((0., 60.), (0., 70.))
    view.plot.setRange(xRange=bounds[0], yRange=bounds[1], padding=0)
    view._manual_view_range_changed(None)
    saved=np.asarray(view.plot.viewRange())
    view.focus_z(.1)
    np.testing.assert_allclose(view.plot.viewRange(), saved)
    switch(view, "position")
    np.testing.assert_allclose(view.plot.viewRange(), original)
    switch(view, mode)
    np.testing.assert_allclose(view.plot.viewRange(), saved)
    view._fit_beam_view()
    assert not np.allclose(view.plot.viewRange(), saved)


def test_intensity_uses_all_rays_not_display_count_and_no_viewport_renormalization(view):
    res=make_result(3001); inc=res.simulation.incident
    inc.ray_weight=np.arange(1, 3002, dtype=float); inc.ray_weight/=inc.ray_weight.sum()
    inc.blocked_z[::3]=.5
    view.display_result(res)
    expected=inc.ray_weight[np.isnan(inc.blocked_z)].sum()
    assert len(view._scatter.data) < 2000
    switch(view, "intensity")
    payload=view.analysis._hover_payload
    assert payload[1].sum() == pytest.approx(expected*200.)
    images=[item for item in view.plot.items() if isinstance(item, pg.ImageItem)]
    assert len(images) == 1 and images[0].axisOrder == "row-major"
    np.testing.assert_array_equal(images[0].image, payload[1].T)
    view.plot.setRange(xRange=(-.02, .02), yRange=(-.02, .02), padding=0)
    view._manual_view_range_changed(None)
    assert view.analysis._hover_payload[1].sum() < expected*200.
    assert view.analysis.plane_data().total_source_fraction == pytest.approx(expected)
    assert "before its detector response" in view.summary.toolTip()


def test_cache_is_shared_between_modes_but_new_publication_invalidates(view, monkeypatch):
    res=make_result(); calls=[]; sample=beam_analysis.sample_beam_plane
    def record(*args):
        calls.append(args); return sample(*args)
    monkeypatch.setattr(beam_analysis, "sample_beam_plane", record)
    view.display_result(res)
    for mode in ("angular", "intensity", "interactions", "angle_histogram", "phase_u"):
        switch(view, mode)
    interaction_colours(view); view.set_projection_angle(90.)
    assert len(calls) == 1
    view.focus_z(.2); assert len(calls) == 2
    view.display_result(res); assert len(calls) == 3


def test_missing_current_uses_source_fraction_and_bad_weights_are_not_fake_intensity(view):
    res=make_result(current=None); view.display_result(res); switch(view, "intensity")
    assert view.analysis._hover_payload[-1] == "% source / bin"
    assert view.analysis._hover_payload[1].sum() == pytest.approx(100.)
    res.simulation.incident.ray_weight[0]=-1.
    view.display_result(res)
    assert view.summary.text() == "Analysis unavailable"
    assert view.analysis._hover_payload is None
    switch(view, "angular")
    assert view._scatter is not None and "Flux unavailable" in view.summary.text()


def test_valid_empty_exit_has_zero_flux_and_no_resurrected_rays(view):
    res=make_result()
    res.specimen_exit=GeometricSpecimenExit((),
        {"tracked_downstream_source_probability":0., "inelastic_absorbed_source_probability":1.}, "current")
    view.display_result(res); view.focus_z(1.5)
    for mode in ("angular", "intensity", "angle_histogram", "interactions"):
        switch(view, mode)
        assert view.analysis.plane_data().ray_count == 0
        assert "Specimen exit" in view.summary.text()
        assert "0%" in view.summary.text()


def test_histogram_hover_reports_physical_angle_and_flux(view):
    view.display_result(make_result()); switch(view, "angle_histogram")
    kind, values, edges, _, unit=view.analysis._hover_payload
    assert kind == "angle" and values.sum() == pytest.approx(200.)
    ix=int(np.argmax(values)); x=float((edges[ix]+edges[ix+1])/2.)
    from PySide6.QtCore import QPointF
    position=view.plot.getViewBox().mapViewToScene(QPointF(x, float(values[ix])/2.))
    view.analysis._mouse_moved(position)
    assert "mrad" in view.analysis.readout.text() and unit in view.analysis.readout.text()


@pytest.mark.parametrize("mode", ["position", "angular", "intensity", "phase_u", "angle_histogram"])
def test_explicit_display_units_never_keep_a_stale_si_tick_multiplier(view, mode, qtbot):
    view.display_result(make_result()); switch(view, mode)
    qtbot.wait(10)
    for name in ("bottom", "left"):
        axis=view.plot.getAxis(name)
        assert axis.autoSIPrefixScale == 1.0
        assert axis.labelUnitPrefix == ""
        assert axis.tickStrings([.1], axis.autoSIPrefixScale * axis.scale, .01) == ["0.10"]


def test_angular_clipping_preserves_fixed_sampling(view):
    res=make_result(3001); res.simulation.incident.blocked_z[::3]=.5
    view.display_result(res); view.focus_z(.1); switch(view, "angular")
    before=view._display_source_ids.copy(); view.focus_z(.9)
    np.testing.assert_array_equal(view._display_source_ids, before[before%3 != 0])


def test_intensity_rebins_resized_view_without_resampling_plane(view, qtbot, monkeypatch):
    view.display_result(make_result()); switch(view, "intensity")
    cached=view.analysis.plane_data()
    def forbidden(*_args):
        pytest.fail("Resizing must only rebin the existing selected-plane cache")
    monkeypatch.setattr(beam_analysis, "sample_beam_plane", forbidden)
    view.resize(510, 900); qtbot.wait(20)
    assert view.analysis.plane_data() is cached
    payload=view.analysis._hover_payload
    np.testing.assert_allclose(payload[2][[0,-1]], view.plot.viewRange()[0])
    np.testing.assert_allclose(payload[3][[0,-1]], view.plot.viewRange()[1])
