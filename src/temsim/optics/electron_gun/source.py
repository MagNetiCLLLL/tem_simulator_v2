"""Single model dispatch for column and alignment source particles."""


def trace_source_to_exit(state, count=None):
    gun = state.electron_gun
    if getattr(gun, "source_representation", "classical_particles") == "effective_gaussian_schell":
        from temsim.optics.electron_gun.effective_source import trace_effective_source
        return trace_effective_source(state, count)
    if count is None:
        return gun.trace_to_exit()
    return gun.trace_to_exit(count)
