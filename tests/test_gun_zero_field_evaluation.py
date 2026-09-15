"""Skip only exactly zero drive; never skip a live alignment or blanker."""
import numpy as np
import pytest

from temsim.optics.electron_gun.alignment import GunDeflector


def coil():
    return GunDeflector(410.,24.,50.,10.,404.,416.,8.)


def test_exact_zero_coils_do_not_evaluate_envelopes(monkeypatch):
    from temsim.optics.electron_gun import alignment
    def unwanted(*args):
        pytest.fail("Unpowered coil evaluated its envelope")
    monkeypatch.setattr(alignment,"_soft_window_with_derivatives",unwanted)
    positions = np.array([[0.,0.,.404],[.001,.002,.416]])
    np.testing.assert_array_equal(coil().field_at_global_positions_t(positions),np.zeros_like(positions))


@pytest.mark.parametrize("blanked",[False,True])
def test_any_nonzero_drive_and_blanking_retain_the_exact_field(blanked):
    from temsim.optics.electron_gun.electrostatic import _soft_window_with_derivatives
    c = coil()
    c.upper_field_x_mt,c.lower_field_y_mt = 1e-25,-.3
    c.beam_blanked = blanked
    positions = np.column_stack((np.zeros((301,2)),np.linspace(.39,.43,301)))
    expected = np.zeros_like(positions)
    if blanked:
        c.enabled = False  # blanking is physically active independently
        fields = [(404.,0.,c.blanking_field_y_mt)]
    else:
        fields = [(404.,1e-25,0.),(416.,0.,-.3)]
    for center,x,y in fields:
        envelope = _soft_window_with_derivatives(positions[:,2]*1000,center-4.,center+4.,1.)[0]
        expected[:,0] += x*1e-3*envelope
        expected[:,1] += y*1e-3*envelope
    np.testing.assert_array_equal(c.field_at_global_positions_t(positions),expected)
