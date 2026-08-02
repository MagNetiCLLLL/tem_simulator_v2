from dataclasses import dataclass


def available_electron_gun_types():
    from temsim.optics.electron_gun import registered_electron_gun_types
    return registered_electron_gun_types()

CORRECTOR_MODES = (
    "probe_corrector",
    "image_corrector",
    "double_corrector",
    "no_corrector",
)

ENERGY_FILTER_MODES = ("energy_filter", "no_energy_filter")

COLUMN_MODES = ("three_lens", "two_lens_c3_off")


def canonical_corrector_mode(mode):
    """Migrate the former dual_corrector spelling at the state boundary."""

    return "double_corrector" if mode == "dual_corrector" else mode


def corrector_mode_for_hardware(mode, c3_hardware):
    """Return the only corrector topology supported by the hardware."""

    mode = canonical_corrector_mode(mode)
    if c3_hardware == "two_condenser":
        return "no_corrector"
    return mode


@dataclass

class TEMConfiguration:

    electron_gun_type: str = "cold_feg"

    corrector_mode: str = "probe_corrector"

    energy_filter_mode: str = "energy_filter"

    monochromator_installed: bool = False


    def validate(self):

        if self.electron_gun_type not in available_electron_gun_types():
            raise ValueError(
                f"Unsupported electron-gun type: {self.electron_gun_type}"
            )

        self.corrector_mode = canonical_corrector_mode(self.corrector_mode)

        if self.corrector_mode not in CORRECTOR_MODES:

            raise ValueError(f"Unsupported corrector mode: {self.corrector_mode}")

        if self.energy_filter_mode not in ENERGY_FILTER_MODES:

            raise ValueError(f"Unsupported energy filter mode: {self.energy_filter_mode}")

        if self.electron_gun_type != "cold_feg":
            self.monochromator_installed = False

        return self



def current_configuration(state):

    return TEMConfiguration(

        state.electron_gun.type_key,

        corrector_mode_for_hardware(
            getattr(state, "corrector_mode", "probe_corrector"),
            getattr(state, "layout_c3_hardware", "three_condenser"),
        ),

        getattr(state, "energy_filter_mode", "energy_filter"),

        bool(getattr(state, "monochromator_installed", False)),

    ).validate()
