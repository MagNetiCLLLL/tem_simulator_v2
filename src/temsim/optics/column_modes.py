"""Public condenser-mode command.

Mode physics belongs to the single state-owned :class:`CondenserSystem`.
This function remains as a small UI-facing command boundary.
"""


def apply_column_mode(state, mode):
    return state.condenser_system.apply_mode(mode)
