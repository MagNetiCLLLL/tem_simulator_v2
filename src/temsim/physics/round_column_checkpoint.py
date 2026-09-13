"""Execute and cache the round prefix before the ordinary 2-D column stage."""
from dataclasses import asdict, replace
import math

from temsim.optics.electron_gun.tip_coherence import wavelength_m
from temsim.physics.radial_cartesian_handoff import RadialColumnNumerics, radial_to_cartesian
from temsim.physics.radial_column_wave import round_column_prefix
from temsim.physics.tip_gun_wave import TipGunCheckpoint, _momentum_velocity
from temsim.physics.wave_flux import BeamState
from temsim.physics.wave_grid import WaveGridNumerics


def execute_round_prefix(state, checkpoint, stop_z_mm, *, maximum_step_mm=.5,
                         numerics=RadialColumnNumerics(), grid_numerics=WaveGridNumerics(),
                         cancelled=lambda: False, progress_callback=None):
    """Only consume the executed gun's matching radial complex expansion.

    The original mode identity, source weight, energy, scattering history and
    phase/clock references survive. A remaining non-axisymmetric operation is
    owned by the subsequent 2-D propagation, never approximated here.
    """
    numerics.validate(); grid_numerics.validate()
    z, radial_modes, record = round_column_prefix(state, checkpoint, stop_z_mm,
        maximum_step_mm=maximum_step_mm, initial_samples=numerics.initial_samples,
        maximum_samples=numerics.maximum_samples, backend=numerics.backend,
        cancelled=cancelled, progress_callback=progress_callback)
    if not radial_modes:
        if z != checkpoint.plane_z_mm:
            raise ValueError("An empty round prefix cannot advance a physical checkpoint")
        return checkpoint
    if not checkpoint.plane_z_mm < z <= stop_z_mm:
        raise ValueError("Round prefix returned an invalid downstream plane")
    if tuple(mode.mode_id for mode, _ in radial_modes) != tuple(mode.mode_id for mode in checkpoint.beam.modes):
        raise ValueError("Round prefix changed the executed source mode identities")
    outputs, rows = [], []
    retained = sum(wave.radius_m.nbytes+wave.amplitude.nbytes for _, wave in radial_modes)
    length_m = (z-checkpoint.plane_z_mm)*1e-3
    for mode, radial in radial_modes:
        if cancelled():
            raise InterruptedError("Round prefix checkpoint cancelled")
        plane, handoff = radial_to_cartesian(radial, wavelength_m=float(wavelength_m(mode.energy_kev*1000)),
            numerics=numerics, grid_numerics=grid_numerics, retained_bytes=retained,
            cancelled=cancelled, progress_callback=progress_callback)
        norm = plane.probability
        momentum, velocity = _momentum_velocity(mode.energy_kev*1000)
        reference = (None if mode.axial_reference is None else
                     mode.axial_reference.advance(float(length_m/velocity), float(length_m*momentum)))
        # Factor a unit shape and absolute weight; their product is exactly
        # the returned field. This does NOT restore the lost/numerical norm.
        output = replace(mode, plane=replace(plane, amplitude=plane.amplitude/math.sqrt(norm) if norm else plane.amplitude),
                         weight_per_reference_electron=mode.weight_per_reference_electron*norm,
                         axial_reference=reference)
        outputs.append(output)
        retained += output.plane.amplitude.nbytes
        rows.append({"mode_id": mode.mode_id, "handoff": handoff,
            "input_weight": mode.weight_per_reference_electron,
            "output_weight": output.weight_per_reference_electron,
            "reference_flight_time_increment_s": float(length_m/velocity),
            "reference_longitudinal_action_increment_j_s": float(length_m*momentum),
            "axial_reference": None if reference is None else asdict(reference)})
    if cancelled():
        raise InterruptedError("Round prefix checkpoint cancelled before publication")
    return TipGunCheckpoint(BeamState(tuple(outputs), checkpoint.beam.reference_plane), z,
        checkpoint.reference_current_a, {"schema": "executed-round-column-checkpoint-v1",
            "upstream_digest": checkpoint.digest, "upstream": checkpoint.record,
            "round_execution": record, "modes": rows, "numerics": asdict(numerics),
            "grid_numerics": grid_numerics.column_identity(),
            "validation_status": "DEVELOPMENT_NOT_SOURCE_ADMISSION"})
