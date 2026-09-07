"""Read-only review reproduction using a synthetic tiny STEM fixture.

Run: .venv/Scripts/python.exe tmp/project_inspection_repro_stem.py
This calls the real wave solver on a 32 x 32 vacuum grid and a 1 x 2 raster.
It does not patch project source or replace any physics function.
"""
from types import SimpleNamespace

import numpy as np

from temsim.optics.column import default_state
from temsim.physics.first_order import TransverseTransfer
from temsim.physics.record_plane import PlaneStop, RecordPlanePlan
from temsim.physics.stem_wave_imaging import (
    AngularDetector,
    simulate_angle_resolved_stem,
)


def main():
    state = default_state()
    state.acceleration_enabled = False
    state.illumination_mode = "STEM"
    state.sample.specimen_preset_key = "vacuum"
    state.sample.thickness_nm = 0.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = False
    state.sample.wave_atomistic_enabled = False
    incident = SimpleNamespace(
        alive=np.ones(5, dtype=bool),
        ray_weight=np.array([0.6, 0.1, 0.1, 0.1, 0.1]),
        x=np.zeros((1, 5)),
        y=np.zeros((1, 5)),
        tx=np.array([[0, 0.002, -0.002, 0, 0]]),
        ty=np.array([[0, 0, 0, 0.002, -0.002]]),
    )
    plane = PlaneStop(
        key="bf", name="BF", kind="detector", z_mm=10.0,
        geometry="disk", outer_width_mm=20.0, readout_enabled=True,
    )
    transfer = TransverseTransfer(
        0.0, 10.0, np.eye(2), np.eye(2), np.zeros((2, 2)), np.eye(2),
    )
    plan = RecordPlanePlan(
        source_z_mm=0.0, planes=(plane,), transfers=(transfer,),
        resolved_geometry_fingerprint="b" * 64, fingerprint="a" * 64,
    )
    scan_x = np.array([[0.0, 0.0001]])
    scan_y = np.zeros_like(scan_x)
    args = (state, SimpleNamespace(incident=incident),
            (AngularDetector("bf", 0.0, 10.0),), scan_x, scan_y)

    try:
        simulate_angle_resolved_stem(*args, record_plane_plan=plan)
    except IndexError as error:
        print("REPRODUCED: record-plane plan with no detector shifts:", error)
    else:
        raise AssertionError("The reviewed IndexError no longer reproduces")

    # The detector is equivalent to the 0-10 mrad angular band. A 100 mrad
    # displacement should move this vacuum probe outside the BF acceptance.
    for use_plan in (False, True):
        results = []
        for shift in (0.0, 100.0):
            result = simulate_angle_resolved_stem(
                *args,
                record_plane_plan=plan if use_plan else None,
                detector_center_shifts_mrad={
                    "bf": (np.full_like(scan_x, shift), scan_y),
                },
            )
            results.append(result.fractions["bf"])
        print("record_plane_plan=", use_plan,
              "BF fractions at 0/100 mrad shift:", results)
        if use_plan:
            assert np.array_equal(*results)
        else:
            assert not np.allclose(*results)


if __name__ == "__main__":
    main()
