"""Bind executed gun fields to the resolved whole-instrument inputs.

The downstream conducting liner is assigned ground in this simulator model.
It is not an additional particle source or an instruction to clamp its energy.
Magnetic captures include every installed contributor without a second gun sum.
"""

from contextlib import contextmanager


@contextmanager
def instrument_gun_field_context(state):
    """Bind shared E-domain and B inputs for an executed gun call or identity.

    The binding is deliberately scoped. A later standalone gun edit must not
    retain a field captured from an earlier whole-instrument state. The field
    includes the gun magnets and installed velocity selector already; callers
    must not add the gun-local magnetic provider a second time.
    """
    from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field
    from temsim.physics.instrument_electric import instrument_electric_end_mm, configure_instrument_electric_domain
    gun = state.electron_gun
    end_mm = instrument_electric_end_mm(state)
    missing = object()
    names = ("_instrument_magnetic_field", "_instrument_magnetic_identity", "_instrument_magnetic_supports_mm",
             "_instrument_magnetic_query_upper_m", "_instrument_electric_end_mm")
    previous = {name: getattr(gun, name, missing) for name in names}
    try:
        configure_instrument_electric_domain(gun, end_mm)
        provider = capture_instrument_magnetic_field(state)
        from temsim.physics.gun_transport_domain import gun_magnetic_query_upper_m
        upper = gun_magnetic_query_upper_m(gun)
        identity = _gun_magnetic_dependency_identity(gun, provider)
        supports = tuple(
            (float(source.bounds_m[0, 2])*1000., float(source.bounds_m[1, 2])*1000.)
            for source in provider._sources
            if not source.known_zero and float(source.bounds_m[0, 2]) <= upper)
        gun._instrument_magnetic_field = provider
        gun._instrument_magnetic_identity = identity
        gun._instrument_magnetic_supports_mm = supports
        gun._instrument_magnetic_query_upper_m = upper
        yield provider
    finally:
        for name, value in previous.items():
            if value is missing:
                if hasattr(gun, name):
                    delattr(gun, name)
            else:
                setattr(gun, name, value)


def _gun_magnetic_dependency_identity(gun, provider):
    """Bind every B contributor inside the enforced gun trial envelope.

    The scalar and compiled gun loops cap dt by the light-cone distance to
    this upper bound. The full provider is still evaluated, with all overlap
    tails intact. Include every upstream source; below their minimum support
    the same finite-support model is identically zero. No electric solve or
    change to its full-instrument boundary is needed for this dependency.
    """
    from temsim.physics.gun_transport_domain import gun_magnetic_query_upper_m
    upper = gun_magnetic_query_upper_m(gun)
    lower = min(0., min((float(source.bounds_m[0, 2])
                        for source in provider._sources), default=0.))
    return provider.identity_for_axial_range(lower, upper)


def gun_transport_magnetic_field(gun):
    """Use the exact bound instrument B; standalone guns retain local B."""
    provider = getattr(gun, "_instrument_magnetic_field", None)
    return gun.magnetic_field if provider is None else provider


def bind_gun_field_environment(gun, assembly):
    if gun.type_key != "cold_feg":
        return
    module = next(item for item in assembly.modules if item.type == "gun")
    settings = module.geometry
    if settings.get("gun_electric_field_model") != "axisymmetric_electrode_laplace":
        raise ValueError("Cold FEG requires an explicit axisymmetric electrode field model")
    if settings.get("downstream_liner_electrical_boundary") != "ground":
        raise ValueError("Cold FEG requires an explicit grounded downstream liner")
    # Keep all physical rows: the field request selects the connected enclosure.
    # Changes to the column can therefore correctly invalidate an upstream solve.
    gun._grounded_outlet_liner_segments = tuple(assembly.vacuum_liner_segments)
    gun._gun_field_cells_per_bore = int(settings["gun_field_cells_per_bore"])
    gun._gun_field_exit_extension_mm = float(settings["gun_field_exit_extension_mm"])
    gun._wien_housing_voltage_reference = settings.get("wien_housing_voltage_reference")
    if gun.monochromator_installed and gun._wien_housing_voltage_reference != "gun_lens":
        raise ValueError("Installed Wien housing requires an explicit gun-lens electrical reference")


def ensure_gun_field_environment(gun):
    """Direct gun callers use a resolved assembly; bound/custom geometry wins."""
    if gun.type_key != "cold_feg" or hasattr(gun, "_grounded_outlet_liner_segments"):
        return
    from temsim.column.layout import LayoutConfiguration
    from temsim.column.module_assembly import resolve_module_assembly
    configuration = LayoutConfiguration(
        electron_gun_type=gun.type_key,
        monochromator_installed=gun.monochromator_installed,
        gun_components=gun.components,
    )
    bind_gun_field_environment(gun, resolve_module_assembly(configuration))
