"""Single model dispatch for column and alignment source particles."""


def trace_source_to_exit(state, count=None):
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    gun = state.electron_gun
    require_physical_gun_source(gun)
    from temsim.vacuum import bind_gun_environment
    bind_gun_environment(state)
    if count is None:
        return gun.trace_to_exit()
    return gun.trace_to_exit(count)
