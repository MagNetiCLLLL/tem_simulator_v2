"""Complete-energy output bookkeeping, not full-source convergence evidence."""
from dataclasses import replace
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits


def test_actual_energy_outputs_survive_later_interruption_and_keep_full_state(tmp_path):
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference, SurfaceCoherence
    from temsim.physics.surface_gun_wave import build_surface_gun_checkpoint
    from temsim.physics.surface_wave import SurfaceWaveNumerics
    from temsim.physics.radial_gun_wave import RadialGunNumerics
    from scripts.surface_mode_evidence import preserve_mode
    gun = default_state().electron_gun
    gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    saved = []
    def output(mode, near, record, radial, history, identity):
        for values in history.values():
            assert not values.flags.writeable
        receipt = preserve_mode(tmp_path, mode, near, record, radial, history, identity)
        with np.load(tmp_path/receipt["file"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["exit_amplitude"], mode.plane.amplitude)
            np.testing.assert_array_equal(data["near_amplitude"], near.amplitude)
            np.testing.assert_array_equal(data["history_coefficients"], history["coefficients"])
            np.testing.assert_array_equal(data["history_covariant_derivatives"], history["covariant_derivatives"])
            assert len(history["z_nm"]) == len(history["coefficients"])
            assert history["chart_derivatives"].shape == (len(history["z_nm"]), 3)
            np.testing.assert_array_equal(data["history_chart_derivatives"], history["chart_derivatives"])
            from temsim.physics.radial_wave_observables import observe_executed_boundary
            for row in record["boundary_states"]:
                j = row["boundary_index"]
                observed = observe_executed_boundary(history, j, np.linspace(0., 4*row["width_nm"], 64),
                    reference_k_per_nm=row["reference_k_per_nm"])
                assert observed.integrated_axial_flux_fraction == pytest.approx(row["net_current_fraction"], abs=1e-13)
                np.testing.assert_array_equal(history["chart_derivatives"][j], row["chart_derivatives"])
            assert len(history["z_nm"]) > len(record["boundary_states"])
            metadata = json.loads(str(data["metadata_json"]))
            assert metadata["identity"]["dependency_digest"] == identity["dependency_digest"]
            assert metadata["mode"]["axial_reference"]
            assert metadata["near"]["phase_reference"]
            assert metadata["scope"] == "COMPLETE_SINGLE_ENERGY_NOT_COMPLETE_SOURCE_OR_IMAGE_ACCEPTANCE"
        with pytest.raises(FileExistsError):
            preserve_mode(tmp_path, mode, near, record, radial, history, identity)
        saved.append(receipt)
        raise InterruptedError("test interruption after a completed energy")
    with threadpool_limits(1), pytest.raises(InterruptedError, match="test interruption"):
        build_surface_gun_checkpoint(gun,
            surface=SurfaceWaveNumerics(element_order=2, radial_nodes=97, axial_nodes=65, outer_radius_factor=4.),
            radial=RadialGunNumerics(radial_modes=8, potential_quadrature=32,
                                     relative_axial_step=.1, field_step_mm=.5),
            _mode_completed=output)
    assert len(saved) == 1
    assert (tmp_path/saved[0]["file"]).is_file()
    assert not list(tmp_path.glob("pending-*"))
