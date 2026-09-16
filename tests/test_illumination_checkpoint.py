import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.optics.assembly_illumination import seed_illumination
from temsim.optics.illumination_checkpoint import IlluminationCheckpoint
from temsim.optics.surface_probe_focus import measure_surface_focus


def test_executed_prefix_matches_full_transport_and_preserves_caller():
    state = default_state()
    seed_illumination(state, 'nano_probe')
    state.electron_gun.emitter.ray_count = 49
    original = state.to_dict()
    keys = ('mini_condenser', 'objective_lens')
    cache = IlluminationCheckpoint(state, keys, step_mm=.05)
    assert state.electron_gun.exit_plane_z_mm < cache.prefix_z_mm < state.sample.z_mm
    lenses = {l.key: l for l in state.lenses}
    initial = np.array([lenses[k].percent for k in keys])
    for shift in ((0., 0.), (.1, .01)):
        values = initial + shift
        cached = cache.measure(values)
        assert state.to_dict() == original
        for k, v in zip(keys, values):
            lenses[k].percent = float(v)
        full = measure_surface_focus(state, step_mm=.05)
        assert cached.statistics.surviving_rays == full.statistics.surviving_rays
        assert cached.statistics.surviving_fraction == pytest.approx(full.statistics.surviving_fraction, abs=1e-12)
        assert cached.statistics.convergence_95_mrad == pytest.approx(full.statistics.convergence_95_mrad, rel=1e-7)
        assert cached.statistics.radius_95_m == pytest.approx(full.statistics.radius_95_m, rel=1e-4, abs=1e-12)
        assert cached.local_waist_offset_nm == pytest.approx(full.local_waist_offset_nm, abs=.1)
        for k, v in zip(keys, initial):
            lenses[k].percent = float(v)
    with pytest.raises(ValueError, match='Finite'):
        cache.measure([np.nan, 40.])
    with pytest.raises(ValueError, match='outside'):
        cache.measure([101., 40.])
