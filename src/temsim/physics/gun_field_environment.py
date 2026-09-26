"""Bind electrical enclosure geometry from the same resolved assembly as optics.

The downstream conducting liner is assigned ground in this simulator model.
It is not an additional particle source or an instruction to clamp its energy.
"""


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
