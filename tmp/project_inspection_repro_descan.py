"""Small read-only comparison of record-plane transfer and real descan kicks."""
import numpy as np

from temsim.optics.column import default_state
from temsim.physics.core import propagate
from temsim.physics.first_order import trace_transverse_transfer


state = default_state()
state.acceleration_enabled = False
state.simulation_mode = "ideal"
state.step_mm = 10.0
descan = state.descan_deflector
descan.enabled = True
descan.scan_enabled = False
source = float(state.sample.z_mm)
target = float(descan.lower_z_mm) + 10.0
offsets = []
physical_positions = []
for kick in (0.0, 1.0):
    descan.kick_x_mrad = kick
    transfer = trace_transverse_transfer(
        state, source, target, maximum_step_mm=10.0,
    )
    events = tuple(descan.kick_events(time_s=0.0))
    zeros = np.array([0.0])
    z, x, tx, y, ty = propagate(
        state, source, target, zeros, zeros, zeros, zeros, events,
        include_spherical_aberration=False, include_hexapole=False,
        maximum_step_mm=10.0,
    )
    offsets.append(transfer.position_offset_m)
    physical_positions.append((x[-1, 0], y[-1, 0]))
    print("descan_kick_x_mrad=", kick, "events=", events,
          "record_plane_transfer_offset_m=", offsets[-1],
          "propagated_position_m=", physical_positions[-1])
assert np.array_equal(*offsets)
assert not np.allclose(*physical_positions, rtol=0.0, atol=1e-12)
