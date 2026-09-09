"""Source colour continuity, including compact finite-specimen descendants."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.diagnostic_tabs import TransverseBeamView, InitialDirectionColourWheel
from temsim.gui.beam_display_source import downstream_display_branches
from temsim.physics.ray_identity import source_identity, branch_identity, select_identity
from temsim.specimen.downstream_transport import GeometricSpecimenExit


def branch(name, z, x, y, **kwargs):
    return SimpleNamespace(name=name, z=np.asarray(z, float), x=np.asarray(x, float),
        y=np.asarray(y, float), blocked_z=np.full(len(x[0]), np.nan), weight=1., **kwargs)


def result():
    theta = np.linspace(0., 2*np.pi, 12, endpoint=False)
    source_x, source_y = np.cos(theta)*1e-6, np.sin(theta)*1e-6
    end_x, end_y = np.cos(theta + .7)*1e-6, np.sin(theta + .7)*1e-6
    incident = branch("incident", [0., 1.], [source_x, end_x], [source_y, end_y])
    post = branch("000", [1., 2.], [end_x, end_x], [end_y, end_y])
    return SimpleNamespace(simulation=SimpleNamespace(incident=incident, branches={"000":post}),
                          signatures={"sample_downstream":"current"})


def displayed(view):
    return [(int(item["source_ray_id"]), brush.color().rgba())
            for item, brush in zip(view._scatter.data["data"], view._scatter.data["brush"])]


def test_source_identity_is_fixed_immutable_and_not_velocity_or_clipping():
    res = result(); inc = res.simulation.incident
    gun = SimpleNamespace(exit_bundle=SimpleNamespace(ray_id=np.arange(12)*7+100))
    ids, angles = source_identity(inc, gun)
    inc.blocked_z[:5] = .5
    inc.tx = np.ones_like(inc.x)*99
    ids2, angles2 = source_identity(inc, gun)
    np.testing.assert_array_equal(ids, ids2)
    np.testing.assert_array_equal(angles, angles2)
    # Azimuth is periodic: roundoff around +X may produce either 0 or 2*pi.
    np.testing.assert_allclose(np.exp(1j*angles),
        np.exp(1j*np.linspace(0, 2*np.pi, 12, endpoint=False)), atol=1e-14)
    with pytest.raises(ValueError):
        ids.setflags(write=True)
    selected_ids, selected_angles = select_identity(ids, angles, np.array([8, 2, 8]))
    np.testing.assert_array_equal(selected_ids, ids[[8,2,8]])
    np.testing.assert_array_equal(selected_angles, angles[[8,2,8]])


def test_vacuum_boundary_preserves_each_source_colour(qtbot):
    res = result(); view = TransverseBeamView(); qtbot.addWidget(view)
    view.display_result(res)
    before = displayed(view)
    xy = np.column_stack([view._scatter.data["x"], view._scatter.data["y"]])
    view.focus_z(1.000001)
    assert displayed(view) == before
    np.testing.assert_array_equal(xy, np.column_stack([view._scatter.data["x"], view._scatter.data["y"]]))
    view.set_projection_angle(90.)
    assert displayed(view) == before
    tip = view._scatter.opts["tip"](1., 2., {"source_ray_id": 4})
    assert "Y 1 µm" in tip and "-X 2 µm" in tip
    assert "Optical reference" in view.summary.text()


def test_clipping_does_not_recentre_downstream_source_colours(qtbot):
    res = result(); view=TransverseBeamView(); qtbot.addWidget(view); view.display_result(res)
    view.focus_z(1.1); before=dict(displayed(view))
    res.simulation.branches["000"].blocked_z[:5] = .5
    view.focus_z(1.2)
    assert dict(displayed(view)) == {i:before[i] for i in range(5,12)}


def test_detailed_exit_uses_all_groups_and_inherits_sparse_repeated_parent_ids(qtbot):
    res=result(); inc=res.simulation.incident
    ids, angles=source_identity(inc)
    children=[]
    for number, parents in enumerate(([8,2,8], [5,9])):
        child_ids, child_angles=select_identity(ids,angles,np.asarray(parents))
        x=np.asarray(parents)*2e-6; y=np.zeros_like(x)
        child=branch(str(number),[1.,2.],[x,x],[y,y],source_ray_id=child_ids,source_azimuth_rad=child_angles)
        child.weight=.4; children.append(child)
    res.specimen_exit=GeometricSpecimenExit(tuple(children),
        {"tracked_downstream_source_probability":.8,"inelastic_absorbed_source_probability":.2},"current")
    view=TransverseBeamView(); qtbot.addWidget(view); view.display_result(res)
    source_colours=dict(displayed(view)); view.focus_z(1.1)
    assert displayed(view) == [(i,source_colours[i]) for i in (8,2,8,5,9)]
    assert "Specimen exit" in view.summary.text()
    assert res.simulation.branches["000"].x.shape[1] == 12


def test_valid_empty_exit_never_resurrects_optical_reference(qtbot):
    res=result(); res.specimen_exit=GeometricSpecimenExit((),
        {"tracked_downstream_source_probability":0.,"inelastic_absorbed_source_probability":1.},"current")
    assert downstream_display_branches(res) == ((), "Specimen exit")
    view=TransverseBeamView(); qtbot.addWidget(view); view.display_result(res); view.focus_z(1.1)
    assert view._scatter is None
    assert "no displayed rays" in view.summary.text()


def test_stale_exit_is_not_used_and_compact_unknown_lineage_stays_neutral(qtbot):
    res=result(); compact=branch("legacy",[1.,2.],[[3e-6],[3e-6]],[[4e-6],[4e-6]])
    ids, angles=branch_identity(compact,res.simulation)
    assert ids.tolist() == [-1] and np.isnan(angles[0])
    res.specimen_exit=GeometricSpecimenExit((compact,),
        {"tracked_downstream_source_probability":1.,"inelastic_absorbed_source_probability":0.},"stale")
    assert downstream_display_branches(res)[1] == "Optical reference"
    res.signatures["sample_downstream"]="stale"
    view=TransverseBeamView(); qtbot.addWidget(view); view.display_result(res); view.focus_z(1.1)
    assert displayed(view) == [(-1,InitialDirectionColourWheel.NEUTRAL_COLOUR.rgba())]
    assert "unavailable" in view.summary.text()


def test_z_clipping_does_not_replace_fixed_display_sample(qtbot):
    res=result(); inc=res.simulation.incident
    x=np.linspace(-1e-6,1e-6,3000); y=np.sin(np.arange(3000))*1e-6
    inc=branch("incident",[0.,1.],[x,x],[y,y]); res.simulation.incident=inc
    inc.blocked_z=np.where(np.arange(3000)%3 == 0,.5,np.nan)
    view=TransverseBeamView(); qtbot.addWidget(view); view.display_result(res)
    view.focus_z(.1); before=dict(displayed(view)); view.focus_z(.9)
    assert dict(displayed(view)) == {i:colour for i,colour in before.items() if i%3 != 0}


def test_detached_full_width_exit_cannot_infer_identity_from_coincident_positions():
    res=result(); inc=res.simulation.incident
    inc.x[-1]=0.; inc.y[-1]=0.
    detached=branch("specimen_exit",[1.,2.],np.zeros((2,12)),np.zeros((2,12)))
    ids, angles=branch_identity(detached,res.simulation)
    assert np.all(ids == -1) and np.all(np.isnan(angles))


def test_incomplete_legacy_exit_falls_back_without_crashing():
    res=result()
    res.sample_region=SimpleNamespace(metrics={"sample_downstream_signature":"current"})
    assert downstream_display_branches(res)[1] == "Optical reference"
    res.signatures="invalid"
    assert downstream_display_branches(res)[1] == "Optical reference"


def test_corrupt_duplicate_incident_ids_are_recovered_but_descendants_can_repeat():
    res=result(); inc=res.simulation.incident
    inc.source_ray_id=np.full(12,7,dtype=np.int64)
    inc.source_azimuth_rad=np.zeros(12)
    ids, angles=source_identity(inc)
    np.testing.assert_array_equal(ids,np.arange(12))
    np.testing.assert_array_equal(branch_identity(inc,res.simulation)[0],ids)
    children=select_identity(ids,angles,np.array([7,2,7]))
    np.testing.assert_array_equal(children[0],[7,2,7])
