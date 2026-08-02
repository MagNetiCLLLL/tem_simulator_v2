"""Compatibility access to the TOML-owned Objective assembly."""

from __future__ import annotations


def resolve_upper_objective_package(state, *, probe_installed):
    """Reload and report the selected manifest's Objective coordinates.

    ``probe_installed`` remains in the public signature for older callers.
    It is checked against the selected topology but never used to calculate
    or move a component.
    """

    from temsim.column.module_assembly import apply_column_manifest_geometry
    from temsim.column.state_layout import layout_configuration_from_state

    configuration = layout_configuration_from_state(state)
    selected_probe = configuration.corrector.value in {
        "probe_corrector",
        "double_corrector",
    }
    if bool(probe_installed) != selected_probe:
        raise ValueError(
            "Objective package topology must match the selected Column TOML"
        )
    apply_column_manifest_geometry(state, configuration)
    return dict(state._upper_objective_package_resolved_positions_mm)
