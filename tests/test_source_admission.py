"""No metadata label can replace an executed gun phase chain."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.source_admission import UnsupportedWaveSource, require_gun_wave_source, launch_emittance_audit


def test_t203_origin_label_and_private_array_do_not_admit_a_wave():
    state = default_state()
    state._wave_source_checkpoint = SimpleNamespace(origin="gun", amplitude=np.ones((16, 16)))
    state._wave_source_node = SimpleNamespace(origin="gun")
    with pytest.raises(UnsupportedWaveSource):
        require_gun_wave_source(state, product="TEM")


def test_t201_api_refuses_legacy_pupil_before_specimen_preparation(monkeypatch):
    from temsim.physics import wave_imaging
    from temsim.physics import stem_wave_imaging
    state = default_state()
    def forbidden(*args, **kwargs):
        raise AssertionError("No specimen or independent pupil may be built")
    monkeypatch.setattr(wave_imaging, "prepare_specimen_potentials", forbidden)
    for call in (lambda: wave_imaging.simulate_wave_image(state, None),
                 lambda: stem_wave_imaging.simulate_angle_resolved_stem(state, None, (), [0.], [0.])):
        with pytest.raises(UnsupportedWaveSource, match="gun-to-specimen"):
            call()


def test_launch_audit_is_deterministic_and_does_not_change_gun():
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = default_state()
    before = capture_instrument_snapshot(state).digest
    first = launch_emittance_audit(state.electron_gun)
    assert first == launch_emittance_audit(state.electron_gun)
    assert first["necessary_quantum_covariance_condition"] == "FAIL"
    # A large violation, not a borderline finite-sampling claim.
    assert max(first["quantum_bound_ratio_xy"]) < .02
    assert capture_instrument_snapshot(state).digest == before


def test_exit_source_is_prohibited_rather_than_rebindable():
    from temsim.optics.electron_gun.effective_source import EffectiveGunSource
    state = default_state()
    gun = state.electron_gun
    gun.effective_source = EffectiveGunSource(1e-9)
    gun.source_representation = "effective_gaussian_schell"
    gun.dpa_aperture.radius_mm *= .9
    with pytest.raises(UnsupportedWaveSource, match="Custom exit sources are not permitted"):
        require_gun_wave_source(state, product="TEM")
