from dataclasses import dataclass, asdict, field

from temsim import module_manifest


_DEFAULT_COLUMN_MODULE = "column/C3_ProbeCorrector.toml"
_DEFAULT_GUN_MODULE = "gun/FEG.toml"
_DEFAULT_SAMPLE_PART = module_manifest.part_data(
    _DEFAULT_COLUMN_MODULE,
    "sample",
)
_DEFAULT_SAMPLE_Z_MM = (
    module_manifest.port_z_mm(_DEFAULT_GUN_MODULE, "exit")
    + float(_DEFAULT_SAMPLE_PART["local_center_z_mm"])
)


@dataclass

class Gaussian:

    amplitude: float

    offset: float

    sigma: float


@dataclass

class Lens:

    name: str

    key: str

    z_mm: float

    b0_t: float

    a_mm: float

    percent: float

    max_percent: float = 100.0

    colour: str = "tab:blue"

    gaussian: list = field(default_factory=list)

    enabled: bool = True

    # Optional component-level electron-optical metadata. The distributed
    # axial field remains the active paraxial model.
    cs_mm: float | None = None

    cc_mm: float | None = None

    polarity: int = 1

    normalise_profile_peak: bool = False

    def scale(self):

        return self.b0_t*self.percent/100.0 if self.enabled else 0.0


@dataclass

class Aperture:

    name: str

    key: str

    z_mm: float

    radius_mm: float

    offset_x_mm: float = 0.0

    offset_y_mm: float = 0.0

    enabled: bool = False

    colour: str = "tab:orange"


    @property

    def radius_um(self):

        return self.radius_mm * 1000.0


    @radius_um.setter

    def radius_um(self, value):

        self.radius_mm = float(value) / 1000.0


    @property

    def offset_x_um(self):

        return self.offset_x_mm * 1000.0


    @offset_x_um.setter

    def offset_x_um(self, value):

        self.offset_x_mm = float(value) / 1000.0


    @property

    def offset_y_um(self):

        return self.offset_y_mm * 1000.0


    @offset_y_um.setter

    def offset_y_um(self, value):

        self.offset_y_mm = float(value) / 1000.0


@dataclass

class Stigmator:

    name: str

    key: str

    z_mm: float

    length_mm: float = 8.0

    max_strength_m2: float = 300.0

    strength_x_percent: float = 0.0

    strength_y_percent: float = 0.0

    enabled: bool = True

    colour: str = "tab:purple"


@dataclass

class DeflectorPair:

    name: str

    key: str

    upper_z_mm: float

    lower_z_mm: float

    upper_x_mrad: float = 0.0

    upper_y_mrad: float = 0.0

    lower_x_mrad: float = 0.0

    lower_y_mrad: float = 0.0

    thickness_mm: float = 8.0

    enabled: bool = True

    colour: str = "tab:cyan"


@dataclass

class Sample:

    z_mm: float = _DEFAULT_SAMPLE_Z_MM

    # The axial sample coordinate remains the probe-analysis reference plane
    # when the holder is retracted.  ``inserted`` controls specimen
    # interaction only; it never moves or removes that optical reference.
    inserted: bool = True

    thickness_nm: float = 100.0

    # Finite specimen envelope in the laboratory sample plane.  The beam
    # travels along +Z; X/Y dimensions and the scan origin are independent of
    # the instrument-owned axial sample position.
    size_x_nm: float = 1000.0

    size_y_nm: float = 1000.0

    centre_x_nm: float = 0.0

    centre_y_nm: float = 0.0

    scan_origin_x_nm: float = 0.0

    scan_origin_y_nm: float = 0.0

    # Legacy qualitative Real-sample ray-preview values are retained only so
    # older profiles round-trip. They no longer generate atomic diffraction or
    # diffuse branches; Real scattering is owned by wave/multislice.
    g_inv_nm: float = 5.0

    excitation_error_inv_nm: float = 0.0

    rocking_width_inv_nm: float = 0.12

    diffuse_broadening_mrad: float = 2.0

    # Enables explicit user-defined channels in Virtual sample mode only.
    diffraction_enabled: bool = True

    # Empty/zero values mean "use the default from the specimen TOML".  The
    # state stores only user choices and overrides, never material constants.
    wave_enabled: bool = False

    # ``atomic`` uses a TOML crystal or user CIF in the wave/multislice path.
    # ``virtual`` uses explicit angular channels in the ray detector model.
    specimen_mode: str = "atomic"

    specimen_preset_key: str = ""

    cif_path: str = ""

    specimen_rotation_x_deg: float = 0.0

    specimen_rotation_y_deg: float = 0.0

    specimen_rotation_z_deg: float = 0.0

    # Canonical physical orientation, stored as a unit quaternion (w,x,y,z).
    # The Euler fields above remain compatibility views for pre-V64 states.
    specimen_orientation_quaternion_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)

    zone_axis_uvw: tuple = (0, 0, 1)

    in_plane_axis_uvw: tuple = (1, 0, 0)

    wave_defocus_nm: float = 0.0

    wave_grid_pixels: int = 0

    wave_field_of_view_angstrom: float = 0.0

    # Multislice can use the legacy qualitative 2-D column projection or a
    # TOML-defined, ASE-oriented 3-D crystal with Lobato IAM slice potentials.
    wave_multislice_enabled: bool = True

    wave_slice_thickness_angstrom: float = 2.0

    wave_bandwidth_fraction: float = 2.0 / 3.0

    wave_atomistic_enabled: bool = True

    wave_frozen_phonon_enabled: bool = False

    wave_frozen_phonon_configurations: int = 4

    # Zero means use the material value and provenance in its specimen TOML.
    wave_frozen_phonon_sigma_angstrom: float = 0.0

    # Custom CIF files do not carry a trustworthy displacement model.  This
    # optional element->one-axis RMS table is therefore explicit and is never
    # silently filled with invented values.
    wave_frozen_phonon_sigma_by_element_angstrom: dict = field(
        default_factory=dict
    )

    wave_frozen_phonon_seed: int = 100

    # Angle-resolved STEM frames are acquired explicitly from the detector
    # panel and therefore never run during ordinary lens recalculation.
    stem_wave_enabled: bool = False

    stem_poisson_enabled: bool = False

    stem_poisson_seed: int = 0

    wave_probe_padding_factor: float = 3.0

    # Real-specimen inelastic transport.  Zero-valued plasmon/ionisation MFP
    # and loss-energy fields select the material values in the specimen TOML.
    # The optional "other" and effective-absorption channels are disabled at
    # zero because no universal material-independent value is defensible.
    real_inelastic_enabled: bool = True

    real_plasmon_mean_free_path_nm: float = 0.0

    real_ionisation_mean_free_path_nm: float = 0.0

    real_other_inelastic_mean_free_path_nm: float = 0.0

    real_absorption_mean_free_path_nm: float = 0.0

    real_plasmon_energy_ev: float = 0.0

    real_ionisation_energy_ev: float = 0.0

    real_other_inelastic_energy_ev: float = 50.0

    # A separately reported high-angle completion model.  It is disabled by
    # default and is never blended into reciprocal-space angles already
    # represented by multislice.
    real_high_angle_tail_enabled: bool = False

    real_tail_atomic_number: int = 14

    real_tail_areal_density_atoms_nm2: float = 0.0

    real_tail_screening_angle_mrad: float = 5.0

    real_tail_max_angle_mrad: float = 250.0

    # Explicit virtual-scatterer controls.  The symmetric diffraction spots
    # and isotropic ring are user-defined angular channels, not calculated
    # Rutherford cross-sections or a crystallographic structure factor.
    virtual_diffraction_angle_mrad: float = 5.0

    virtual_diffraction_azimuth_deg: float = 0.0

    virtual_diffraction_relative_weight: float = 1.0

    virtual_scattering_angle_mrad: float = 20.0

    virtual_scattering_relative_weight: float = 0.2

    virtual_scattering_azimuth_samples: int = 16

    # Extensible absolute-probability interaction rows.  The unlisted
    # probability is the direct-beam remainder; enabled rows must never sum
    # above one.  These defaults preserve an inspectable virtual specimen
    # without claiming a crystallographic or Rutherford calculation.
    virtual_interactions: list = field(default_factory=lambda: [
        {
            "name": "Diffraction pair",
            "kind": "diffraction_spots",
            "enabled": True,
            "probability": 0.20,
            "angle_mrad": 5.0,
            "azimuth_deg": 0.0,
            "spot_count": 2,
        },
        {
            "name": "Diffuse ring",
            "kind": "diffuse_ring",
            "enabled": True,
            "probability": 0.05,
            "angle_mrad": 20.0,
            "width_mrad": 2.0,
            "azimuth_samples": 32,
        },
    ])

    # Empty means one uniform finite slab using size_x_nm/size_y_nm.  Region
    # rows can be rectangles, ellipses, or imported grayscale density maps.
    virtual_regions: list = field(default_factory=list)

    virtual_probe_convolution_enabled: bool = True

    @property
    def upper_surface_z_mm(self):
        return self.z_mm - self.thickness_nm * 0.5e-6

    @property
    def lower_surface_z_mm(self):
        return self.z_mm + self.thickness_nm * 0.5e-6


@dataclass

class State:

    lenses: list

    apertures: list

    stigmators: list

    deflectors: list

    electron_gun: object = None
    electron_gun_profiles: dict = field(default_factory=dict, repr=False)

    sample: Sample = field(default_factory=Sample)

    camera: object = None

    # Canonical editable geometry store shared by all modular components.
    component_placements: dict = field(default_factory=dict)

    illumination_mode: str = "STEM"

    projector_mode: str = "diffraction"

    # Enabled after applying a five-lens image-magnification preset.  The
    # engineering model uses field-integral focal lengths plus separate
    # Larmor rotations; diffraction mode always retains distributed fields.
    equivalent_image_lenses_enabled: bool = False

    step_mm: float = 0.5

    history_step_mm: float = 2.0

    acceleration_enabled: bool = True

    acceleration_backend: str = "Auto"

    active_backend: str = "CPU"

    corrector_mode: str = "probe_corrector"

    energy_filter_mode: str = "energy_filter"

    column_mode: str = "three_lens"

    c2c3_crossover_required: bool = True

    objective_coupled: bool = True

    # Derived from the selected TOML assembly; never a persisted geometry input.
    corrector_crossover_targets_mm: list = field(default_factory=list)

    energy_filter: object = None

    layout_c3_hardware: str = "three_condenser"

    layout_c3_excited: bool = True

    layout_reference_positions: dict = field(default_factory=dict, repr=False)
    layout_reference_enabled: dict = field(default_factory=dict, repr=False)
    monochromator_column_offset_mm: float = 0.0
    monochromator_axis_offset_mm: float = 0.0
    ac_downstream_anchor_offsets_mm: dict = field(
        default_factory=dict,
        repr=False,
    )
    mini_condenser_upstream_gap_mm: float = 0.0
    objective_stigmator_symmetry_offset_mm: float = 0.0
    image_diffraction_deflector_upstream_gap_mm: float = 0.0
    image_corrector_upstream_gap_mm: float = 5.0
    image_corrector_component_offsets_from_ol_post_mm: dict = field(
        default_factory=dict,
        repr=False,
    )
    selected_area_aperture_offset_from_sad_mm: float = 0.0
    standalone_selected_area_aperture_gap_after_descan_mm: float = 5.0
    wobble_observation_plane_key: str = "flu_screen"
    wobble_custom_observation_z_mm: float = _DEFAULT_SAMPLE_Z_MM
    virtual_observation_z_mm: float = _DEFAULT_SAMPLE_Z_MM
    chromatic_aberration_enabled: bool = False

    schema_version: int = 64

    def __post_init__(self):
        if self.electron_gun is None:
            from temsim.optics.electron_gun import create_electron_gun
            self.electron_gun = create_electron_gun("cold_feg")

    @property
    def monochromator_installed(self):
        gun = getattr(self, "electron_gun", None)
        return bool(
            gun is not None
            and gun.type_key == "cold_feg"
            and getattr(gun, "monochromator_installed", False)
        )

    @monochromator_installed.setter
    def monochromator_installed(self, value):
        gun = getattr(self, "electron_gun", None)
        can_install = (
            gun is not None
            and gun.type_key == "cold_feg"
            and getattr(gun, "monochromator", None) is not None
        )
        if bool(value):
            if can_install:
                gun.install_monochromator()
                from temsim import module_manifest
                self._set_monochromator_column_offset(
                    module_manifest.port_z_mm(
                        "gun/FEG_Mono.toml", "exit"
                    )
                    - module_manifest.port_z_mm(
                        "gun/FEG.toml", "exit"
                    )
                )
        else:
            if can_install:
                gun.remove_monochromator()
            self._set_monochromator_column_offset(0.0)

    def _set_monochromator_column_offset(self, target_mm):
        """Select the matching Gun TOML and rebuild all downstream positions."""

        target_mm = float(target_mm)
        self.monochromator_column_offset_mm = target_mm
        if not hasattr(self, "corrector_elements"):
            return
        if self.electron_gun.type_key not in {"cold_feg", "thermionic"}:
            return
        from temsim.column.module_assembly import apply_column_manifest_geometry
        from temsim.column.state_layout import layout_configuration_from_state
        apply_column_manifest_geometry(
            self,
            layout_configuration_from_state(self),
        )

    @staticmethod
    def _translate_component_from_tip(component, delta_mm):
        coordinate_groups = (
            (
                "mechanical_center_from_tip_mm",
                (
                    "optical_reference_from_tip_mm",
                    "optical_upper_reference_from_tip_mm",
                    "optical_lower_reference_from_tip_mm",
                ),
            ),
            (
                "standalone_mechanical_center_from_tip_mm",
                ("standalone_optical_reference_from_tip_mm",),
            ),
        )
        for mechanical_name, optical_names in coordinate_groups:
            if not hasattr(component, mechanical_name):
                continue
            optical_before = {
                name: float(getattr(component, name))
                for name in optical_names
                if hasattr(component, name)
            }
            setattr(
                component,
                mechanical_name,
                float(getattr(component, mechanical_name)) + delta_mm,
            )
            # Most modular components couple their optical plane to mechanical
            # placement. Condenser lens views deliberately expose independent
            # fields, so translate any optical coordinate that did not move.
            for name, previous in optical_before.items():
                if abs(float(getattr(component, name)) - previous) <= 1.0e-12:
                    setattr(component, name, previous + delta_mm)

    @property
    def beam_voltage_kv(self):
        return float(self.electron_gun.nominal_exit_energy_ev) / 1000.0

    def replace_electron_gun(self, assembly):
        """Install an explicit complete gun without changing the column."""

        required = (
            "type_key",
            "display_name",
            "components",
            "exit_plane_z_mm",
            "nominal_exit_energy_ev",
            "emitted_current_a",
            "ray_count",
            "diagnostic_waist_region_mm",
            "validate",
            "emit",
            "trace_to_exit",
            "draw_layout",
            "to_dict",
        )
        missing = [
            name for name in required
            if not hasattr(assembly, name)
        ]
        if missing:
            raise TypeError(
                "Replacement gun is missing interface members: "
                + ", ".join(missing)
            )
        assembly.validate()
        previous = getattr(self, "electron_gun", None)
        if previous is not None and previous is not assembly:
            self._set_monochromator_column_offset(0.0)
        if (
            previous is not None
            and previous is not assembly
            and hasattr(previous, "to_dict")
        ):
            self.electron_gun_profiles[previous.type_key] = (
                previous.to_dict()
            )
        self.electron_gun = assembly
        self.electron_gun_profiles.pop(assembly.type_key, None)
        if (
            assembly.type_key == "cold_feg"
            and getattr(assembly, "monochromator_installed", False)
        ):
            from temsim import module_manifest
            self._set_monochromator_column_offset(
                module_manifest.port_z_mm(
                    "gun/FEG_Mono.toml", "exit"
                )
                - module_manifest.port_z_mm(
                    "gun/FEG.toml", "exit"
                )
            )
        else:
            self._set_monochromator_column_offset(0.0)
        return assembly

    def select_electron_gun(self, type_key):
        """Switch gun family while preserving each family's own settings."""

        requested = str(type_key)
        if requested == self.electron_gun.type_key:
            return self.electron_gun
        from temsim.optics.electron_gun import create_electron_gun

        if (
            self.electron_gun.type_key == "cold_feg"
            and requested != "cold_feg"
            and getattr(self.electron_gun, "monochromator", None) is not None
        ):
            # Source-family mutual exclusion removes the hardware while the
            # complete field/slit parameter block remains in the FEG profile.
            self.monochromator_installed = False
        stored = self.electron_gun_profiles.get(requested)
        replacement = create_electron_gun(requested, stored)
        return self.replace_electron_gun(replacement)

    @property
    def condenser_system(self):
        """Return the single cached condenser-system instance for this state."""
        from temsim.optics.condenser_lens import CondenserSystem

        system = getattr(self, "_condenser_system", None)
        if system is None or system.state is not self:
            system = CondenserSystem(self)
            self._condenser_system = system
        return system

    @property
    def probe_corrector_system(self):
        """Return the cached modular probe-corrector assembly."""
        from temsim.optics.probe_corrector import ProbeCorrectorSystem

        system = getattr(self, "_probe_corrector_system", None)
        if system is None or system.state is not self:
            system = ProbeCorrectorSystem(self)
            self._probe_corrector_system = system
        return system

    @property
    def image_corrector_system(self):
        """Return the cached modular image-corrector assembly."""
        from temsim.optics.image_corrector import ImageCorrectorSystem

        system = getattr(self, "_image_corrector_system", None)
        if system is None or system.state is not self:
            system = ImageCorrectorSystem(self)
            self._image_corrector_system = system
        return system

    @property
    def condenser_lens_1(self):
        return self.condenser_system.condenser_lens_1

    @property
    def condenser_lens_2(self):
        return self.condenser_system.condenser_lens_2

    @property
    def condenser_lens_3(self):
        return self.condenser_system.condenser_lens_3

    @property
    def condenser_aperture_2(self):
        """Return the sole continuous C2 aperture component in this state."""
        from temsim.component_keys import CONDENSER_APERTURE_2

        cached = getattr(self, "_condenser_aperture_2", None)
        if (
            cached is None
            or cached not in self.apertures
            or cached.key != CONDENSER_APERTURE_2
        ):
            cached = next(
                aperture
                for aperture in self.apertures
                if aperture.key == CONDENSER_APERTURE_2
            )
            self._condenser_aperture_2 = cached
        return cached

    @property
    def condenser_aperture_3(self):
        """Return the sole continuous C3 aperture component in this state."""
        from temsim.component_keys import CONDENSER_APERTURE_3

        cached = getattr(self, "_condenser_aperture_3", None)
        if (
            cached is None
            or cached not in self.apertures
            or cached.key != CONDENSER_APERTURE_3
        ):
            cached = next(
                aperture
                for aperture in self.apertures
                if aperture.key == CONDENSER_APERTURE_3
            )
            self._condenser_aperture_3 = cached
        return cached

    @property
    def condenser_deflector(self):
        """Return the sole condenser-deflector component in this state."""
        from temsim.component_keys import CONDENSER_DEFLECTOR

        cached = getattr(self, "_condenser_deflector", None)
        if (
            cached is None
            or cached not in self.deflectors
            or cached.key != CONDENSER_DEFLECTOR
        ):
            cached = next(
                deflector
                for deflector in self.deflectors
                if deflector.key == CONDENSER_DEFLECTOR
            )
            self._condenser_deflector = cached
        return cached

    @property
    def adapter_lens(self):
        """Return the sole ADL component in this state."""
        from temsim.component_keys import ADAPTER_LENS

        cached = getattr(self, "_adapter_lens", None)
        if (
            cached is None
            or cached not in self.lenses
            or cached.key != ADAPTER_LENS
        ):
            cached = next(
                lens for lens in self.lenses if lens.key == ADAPTER_LENS
            )
            self._adapter_lens = cached
        return cached

    @property
    def image_corrector_ol_post_lens(self):
        """Return the sole Image Corrector OL-post round lens."""
        from temsim.component_keys import IMAGE_CORRECTOR_OL_POST_LENS

        return self._probe_lens(
            IMAGE_CORRECTOR_OL_POST_LENS,
            "_image_corrector_ol_post_lens",
        )

    @property
    def tl22_lens(self):
        """Return the sole Probe Corrector TL22 component."""
        from temsim.component_keys import PROBE_TL22_LENS

        cached = getattr(self, "_tl22_lens", None)
        if (
            cached is None
            or cached not in self.lenses
            or cached.key != PROBE_TL22_LENS
        ):
            cached = next(
                lens
                for lens in self.lenses
                if lens.key == PROBE_TL22_LENS
            )
            self._tl22_lens = cached
        return cached

    @property
    def dph2_deflector(self):
        """Return the sole Probe Corrector DPH2 component."""
        from temsim.component_keys import PROBE_DPH2_DEFLECTOR

        cached = getattr(self, "_dph2_deflector", None)
        elements = getattr(self, "corrector_elements", [])
        if (
            cached is None
            or cached not in elements
            or cached.key != PROBE_DPH2_DEFLECTOR
        ):
            cached = next(
                item
                for item in elements
                if item.key == PROBE_DPH2_DEFLECTOR
            )
            self._dph2_deflector = cached
        return cached

    @property
    def objective_aperture(self):
        """Return the canonical cartridge and hard-edge objective stop."""
        from temsim.component_keys import OBJECTIVE_APERTURE

        cached = getattr(self, "_objective_aperture", None)
        if (
            cached is None
            or cached not in self.apertures
            or cached.key != OBJECTIVE_APERTURE
        ):
            cached = next(
                aperture
                for aperture in self.apertures
                if aperture.key == OBJECTIVE_APERTURE
            )
            self._objective_aperture = cached
        return cached

    @property
    def selected_area_aperture(self):
        """Return the topology-aware Selected Area Aperture."""
        from temsim.component_keys import SELECTED_AREA_APERTURE

        cached = getattr(self, "_selected_area_aperture", None)
        if (
            cached is None
            or cached not in self.apertures
            or cached.key != SELECTED_AREA_APERTURE
        ):
            cached = next(
                aperture
                for aperture in self.apertures
                if aperture.key == SELECTED_AREA_APERTURE
            )
            self._selected_area_aperture = cached
        return cached

    @property
    def energy_filter_entrance_aperture(self):
        """Return the Camera-anchored Energy Filter Entrance Aperture."""
        from temsim.component_keys import ENERGY_FILTER_ENTRANCE_APERTURE

        cached = getattr(
            self, "_energy_filter_entrance_aperture", None
        )
        if (
            cached is None
            or cached not in self.apertures
            or cached.key != ENERGY_FILTER_ENTRANCE_APERTURE
        ):
            cached = next(
                aperture
                for aperture in self.apertures
                if aperture.key == ENERGY_FILTER_ENTRANCE_APERTURE
            )
            self._energy_filter_entrance_aperture = cached
        return cached

    @property
    def energy_filter_entrance_m12(self):
        """Return the independent pre-sector Energy Filter M12."""
        from temsim.optics.energy_filter import ensure_energy_filter

        ensure_energy_filter(self)
        return self.energy_filter.entrance_m12

    @property
    def energy_filter_exit_m12(self):
        """Return the independent post-sector Energy Filter M12."""
        from temsim.optics.energy_filter import ensure_energy_filter

        ensure_energy_filter(self)
        return self.energy_filter.exit_m12

    @property
    def mini_condenser(self):
        """Return the sole topology-aware Mini Condenser component."""
        from temsim.component_keys import MINI_CONDENSER

        cached = getattr(self, "_mini_condenser", None)
        if (
            cached is None
            or cached not in self.lenses
            or cached.key != MINI_CONDENSER
        ):
            cached = next(
                lens for lens in self.lenses
                if lens.key == MINI_CONDENSER
            )
            self._mini_condenser = cached
        return cached

    @property
    def condenser_stigmator(self):
        """Return the sole shared-column Condenser Stigmator."""
        from temsim.component_keys import CONDENSER_STIGMATOR

        cached = getattr(self, "_condenser_stigmator", None)
        if (
            cached is None
            or cached not in self.stigmators
            or cached.key != CONDENSER_STIGMATOR
        ):
            cached = next(
                stigmator
                for stigmator in self.stigmators
                if stigmator.key == CONDENSER_STIGMATOR
            )
            self._condenser_stigmator = cached
        return cached

    @property
    def objective_stigmator(self):
        """Return the owned continuous Objective Stigmator component."""
        from temsim.component_keys import OBJECTIVE_STIGMATOR

        cached = getattr(self, "_objective_stigmator", None)
        if (
            cached is None
            or cached not in self.stigmators
            or cached.key != OBJECTIVE_STIGMATOR
        ):
            cached = next(
                stigmator
                for stigmator in self.stigmators
                if stigmator.key == OBJECTIVE_STIGMATOR
            )
            self._objective_stigmator = cached
        return cached

    @property
    def diffraction_stigmator(self):
        """Return the independent topology-aware Diffraction Stigmator."""
        from temsim.component_keys import DIFFRACTION_STIGMATOR

        cached = getattr(self, "_diffraction_stigmator", None)
        if (
            cached is None
            or cached not in self.stigmators
            or cached.key != DIFFRACTION_STIGMATOR
        ):
            cached = next(
                stigmator
                for stigmator in self.stigmators
                if stigmator.key == DIFFRACTION_STIGMATOR
            )
            self._diffraction_stigmator = cached
        return cached

    @property
    def diffraction_lens(self):
        """Return the independent topology-aware Diffraction Lens."""
        from temsim.component_keys import DIFFRACTION_LENS

        cached = getattr(self, "_diffraction_lens", None)
        if (
            cached is None
            or cached not in self.lenses
            or cached.key != DIFFRACTION_LENS
        ):
            cached = next(
                lens
                for lens in self.lenses
                if lens.key == DIFFRACTION_LENS
            )
            self._diffraction_lens = cached
        return cached

    @property
    def intermediate_lens(self):
        """Return the Intermediate Lens anchored to the Diffraction Lens."""
        from temsim.component_keys import INTERMEDIATE_LENS

        cached = getattr(self, "_intermediate_lens", None)
        if (
            cached is None
            or cached not in self.lenses
            or cached.key != INTERMEDIATE_LENS
        ):
            cached = next(
                lens
                for lens in self.lenses
                if lens.key == INTERMEDIATE_LENS
            )
            self._intermediate_lens = cached
        return cached

    @property
    def projector_lens_p1(self):
        """Return Projector Lens P1 anchored to the Diffraction Lens."""
        from temsim.component_keys import PROJECTOR_LENS_1

        cached = getattr(self, "_projector_lens_p1", None)
        if (
            cached is None
            or cached not in self.lenses
            or cached.key != PROJECTOR_LENS_1
        ):
            cached = next(
                lens
                for lens in self.lenses
                if lens.key == PROJECTOR_LENS_1
            )
            self._projector_lens_p1 = cached
        return cached

    @property
    def projector_lens_p2(self):
        """Return Projector Lens P2 anchored to the Diffraction Lens."""
        from temsim.component_keys import PROJECTOR_LENS_2

        cached = getattr(self, "_projector_lens_p2", None)
        if (
            cached is None
            or cached not in self.lenses
            or cached.key != PROJECTOR_LENS_2
        ):
            cached = next(
                lens
                for lens in self.lenses
                if lens.key == PROJECTOR_LENS_2
            )
            self._projector_lens_p2 = cached
        return cached

    @property
    def stem_detectors(self):
        """Return the canonical BF, DF and HAADF detector components."""
        from temsim.component_keys import STEM_DETECTOR_KEYS

        planes = getattr(self, "recording_planes", [])
        by_key = {plane.key: plane for plane in planes}
        return tuple(by_key[key] for key in STEM_DETECTOR_KEYS)

    @property
    def haadf_detector(self):
        from temsim.component_keys import HAADF_DETECTOR

        return next(
            detector
            for detector in self.stem_detectors
            if detector.key == HAADF_DETECTOR
        )

    @property
    def dark_field_detector(self):
        from temsim.component_keys import DARK_FIELD_DETECTOR

        return next(
            detector
            for detector in self.stem_detectors
            if detector.key == DARK_FIELD_DETECTOR
        )

    @property
    def bright_field_detector(self):
        from temsim.component_keys import BRIGHT_FIELD_DETECTOR

        return next(
            detector
            for detector in self.stem_detectors
            if detector.key == BRIGHT_FIELD_DETECTOR
        )

    @property
    def fluorescent_screen(self):
        """Return the canonical retractable fluorescent-screen component."""
        from temsim.component_keys import FLUORESCENT_SCREEN

        return next(
            plane
            for plane in self.recording_planes
            if plane.key == FLUORESCENT_SCREEN
        )

    @property
    def objective_lens(self):
        """Return the sole coupled Objective Lens assembly."""
        from temsim.component_keys import OBJECTIVE_LENS

        cached = getattr(self, "_objective_lens", None)
        if (
            cached is None
            or cached not in self.lenses
            or cached.key != OBJECTIVE_LENS
        ):
            cached = next(
                lens
                for lens in self.lenses
                if lens.key == OBJECTIVE_LENS
            )
            self._objective_lens = cached
        return cached

    @property
    def ac_deflector(self):
        """Return the sole shared-column AC wobble deflector."""
        from temsim.component_keys import AC_DEFLECTOR

        cached = getattr(self, "_ac_deflector", None)
        elements = getattr(self, "corrector_elements", [])
        if (
            cached is None
            or cached not in elements
            or cached.key != AC_DEFLECTOR
        ):
            cached = next(
                item for item in elements if item.key == AC_DEFLECTOR
            )
            self._ac_deflector = cached
        return cached

    @property
    def descan_deflector(self):
        """Return the sole independent post-objective Descan Deflector."""
        from temsim.component_keys import DESCAN_DEFLECTOR

        cached = getattr(self, "_descan_deflector", None)
        elements = getattr(self, "corrector_elements", [])
        if (
            cached is None
            or cached not in elements
            or cached.key != DESCAN_DEFLECTOR
        ):
            cached = next(
                item for item in elements
                if item.key == DESCAN_DEFLECTOR
            )
            self._descan_deflector = cached
        return cached

    @property
    def dp22_deflector(self):
        """Return the sole Probe Corrector DP22 component."""
        from temsim.component_keys import PROBE_DP22_DEFLECTOR

        cached = getattr(self, "_dp22_deflector", None)
        elements = getattr(self, "corrector_elements", [])
        if (
            cached is None
            or cached not in elements
            or cached.key != PROBE_DP22_DEFLECTOR
        ):
            cached = next(
                item
                for item in elements
                if item.key == PROBE_DP22_DEFLECTOR
            )
            self._dp22_deflector = cached
        return cached

    @property
    def qph2_quadrupole(self):
        """Return the sole Probe Corrector QPH2 component."""
        from temsim.component_keys import PROBE_QPH2_QUADRUPOLE

        cached = getattr(self, "_qph2_quadrupole", None)
        elements = getattr(self, "corrector_elements", [])
        if (
            cached is None
            or cached not in elements
            or cached.key != PROBE_QPH2_QUADRUPOLE
        ):
            cached = next(
                item
                for item in elements
                if item.key == PROBE_QPH2_QUADRUPOLE
            )
            self._qph2_quadrupole = cached
        return cached

    @property
    def hp2_hexapole(self):
        """Return the sole Probe Corrector HP2 component."""
        from temsim.component_keys import PROBE_HP2_HEXAPOLE

        cached = getattr(self, "_hp2_hexapole", None)
        elements = getattr(self, "corrector_elements", [])
        if (
            cached is None
            or cached not in elements
            or cached.key != PROBE_HP2_HEXAPOLE
        ):
            cached = next(
                item
                for item in elements
                if item.key == PROBE_HP2_HEXAPOLE
            )
            self._hp2_hexapole = cached
        return cached

    @property
    def hpc_hexapole(self):
        """Return the sole Probe Corrector HPC component."""
        from temsim.component_keys import PROBE_HPC_HEXAPOLE

        cached = getattr(self, "_hpc_hexapole", None)
        elements = getattr(self, "corrector_elements", [])
        if (
            cached is None
            or cached not in elements
            or cached.key != PROBE_HPC_HEXAPOLE
        ):
            cached = next(
                item
                for item in elements
                if item.key == PROBE_HPC_HEXAPOLE
            )
            self._hpc_hexapole = cached
        return cached

    def _probe_corrector_element(self, key, cache_attribute):
        cached = getattr(self, cache_attribute, None)
        elements = getattr(self, "corrector_elements", [])
        if cached is None or cached not in elements or cached.key != key:
            cached = next(item for item in elements if item.key == key)
            setattr(self, cache_attribute, cached)
        return cached

    def _probe_lens(self, key, cache_attribute):
        cached = getattr(self, cache_attribute, None)
        if cached is None or cached not in self.lenses or cached.key != key:
            cached = next(item for item in self.lenses if item.key == key)
            setattr(self, cache_attribute, cached)
        return cached

    def _probe_deflector(self, key, cache_attribute):
        cached = getattr(self, cache_attribute, None)
        if (
            cached is None
            or cached not in self.deflectors
            or cached.key != key
        ):
            cached = next(
                item for item in self.deflectors if item.key == key
            )
            setattr(self, cache_attribute, cached)
        return cached

    @property
    def qpc_quadrupole(self):
        from temsim.component_keys import PROBE_QPC_QUADRUPOLE
        return self._probe_corrector_element(
            PROBE_QPC_QUADRUPOLE, "_qpc_quadrupole"
        )

    @property
    def dp21_deflector(self):
        from temsim.component_keys import PROBE_DP21_DEFLECTOR
        return self._probe_corrector_element(
            PROBE_DP21_DEFLECTOR, "_dp21_deflector"
        )

    @property
    def tl21_lens(self):
        from temsim.component_keys import PROBE_TL21_LENS
        return self._probe_lens(PROBE_TL21_LENS, "_tl21_lens")

    @property
    def dph1_deflector(self):
        from temsim.component_keys import PROBE_DPH1_DEFLECTOR
        return self._probe_corrector_element(
            PROBE_DPH1_DEFLECTOR, "_dph1_deflector"
        )

    @property
    def qph1_quadrupole(self):
        from temsim.component_keys import PROBE_QPH1_QUADRUPOLE
        return self._probe_corrector_element(
            PROBE_QPH1_QUADRUPOLE, "_qph1_quadrupole"
        )

    @property
    def hp1_hexapole(self):
        from temsim.component_keys import PROBE_HP1_HEXAPOLE
        return self._probe_corrector_element(
            PROBE_HP1_HEXAPOLE, "_hp1_hexapole"
        )

    @property
    def hpol_hexapole(self):
        from temsim.component_keys import PROBE_HPOL_HEXAPOLE
        return self._probe_corrector_element(
            PROBE_HPOL_HEXAPOLE, "_hpol_hexapole"
        )

    @property
    def qpol_quadrupole(self):
        from temsim.component_keys import PROBE_QPOL_QUADRUPOLE
        return self._probe_corrector_element(
            PROBE_QPOL_QUADRUPOLE, "_qpol_quadrupole"
        )

    @property
    def dp11_deflector(self):
        from temsim.component_keys import PROBE_DP11_DEFLECTOR
        return self._probe_corrector_element(
            PROBE_DP11_DEFLECTOR, "_dp11_deflector"
        )

    @property
    def tl12_lens(self):
        from temsim.component_keys import PROBE_TL12_LENS
        return self._probe_lens(PROBE_TL12_LENS, "_tl12_lens")

    @property
    def dp12_scan_deflector(self):
        from temsim.component_keys import PROBE_DP12_SCAN_DEFLECTOR
        return self._probe_deflector(
            PROBE_DP12_SCAN_DEFLECTOR, "_dp12_scan_deflector"
        )

    @property
    def beam_deflector(self):
        """Return the sole beam shift/tilt deflector in this state."""
        from temsim.component_keys import BEAM_DEFLECTOR

        cached = getattr(self, "_beam_deflector", None)
        if (
            cached is None
            or cached not in self.deflectors
            or cached.key != BEAM_DEFLECTOR
        ):
            cached = next(
                deflector
                for deflector in self.deflectors
                if deflector.key == BEAM_DEFLECTOR
            )
            self._beam_deflector = cached
        return cached

    @property
    def image_diffraction_deflector(self):
        """Return the owned post-objective paired deflector."""
        from temsim.component_keys import IMAGE_DIFFRACTION_DEFLECTOR

        cached = getattr(self, "_image_diffraction_deflector", None)
        if (
            cached is None
            or cached not in self.deflectors
            or cached.key != IMAGE_DIFFRACTION_DEFLECTOR
        ):
            cached = next(
                deflector
                for deflector in self.deflectors
                if deflector.key == IMAGE_DIFFRACTION_DEFLECTOR
            )
            self._image_diffraction_deflector = cached
        return cached


    def sync_objective(self):
        try:
            objective = self.objective_lens.validate()
        except (StopIteration, AttributeError):
            return
        signature = (
            id(objective),
            float(self.beam_voltage_kv),
            float(self.sample.z_mm),
            float(self.sample.thickness_nm),
            float(objective.percent),
            bool(objective.enabled),
            int(getattr(objective, "polarity", 1)),
        )
        if (
            getattr(self, "_objective_plane_signature", None) == signature
            and hasattr(self, "objective_back_focal_plane_z_mm")
            and hasattr(self, "objective_image_plane_z_mm")
        ):
            return
        back_focal_z_mm = objective.back_focal_plane_z_mm(
            self.beam_voltage_kv,
            self.sample,
        )
        image_z_mm = objective.image_plane_z_mm(
            self.beam_voltage_kv,
            self.sample,
        )
        objective._back_focal_plane_z_mm = back_focal_z_mm
        objective._image_plane_z_mm = image_z_mm
        self.objective_back_focal_plane_z_mm = back_focal_z_mm
        self.objective_image_plane_z_mm = image_z_mm
        self._objective_plane_signature = signature


    def to_dict(self):

        """Return a complete, versioned, JSON-safe simulator state."""
        from temsim.component_keys import (
            OBJECTIVE_LENS,
            canonical_component_placement_key,
        )
        from temsim.detector.recording_system import serialise_recording_system
        from temsim.optics.corrector_structure import serialise_corrector_structure
        from temsim.optics.energy_filter import (
            ensure_energy_filter,
            serialise_energy_filter,
        )
        from temsim.configuration import corrector_mode_for_hardware
        from temsim.column.module_assembly import (
            STRUCTURAL_FIELD_SOURCES,
            TOML_OWNED_GEOMETRY_KEYS,
        )

        def strip_position_fields(payload, component_key=None):
            payload = dict(payload)
            component_key = canonical_component_placement_key(component_key)
            for key in tuple(payload):
                if (
                    key in {
                        "z_mm",
                        "upper_z_mm",
                        "lower_z_mm",
                        "mechanical_length_mm",
                        "assembly_length_mm",
                        "optical_plane_separation_mm",
                        "coil_plane_separation_mm",
                        "sample_axial_offset_mm",
                    }
                    or "mechanical_center_" in key
                    or "optical_reference_" in key
                    or "layout_center_" in key
                    or key.startswith("optical_upper_reference_")
                    or key.startswith("optical_lower_reference_")
                    or key.endswith("_field_center_z_mm")
                    or key.endswith("_pole_piece_center_z_mm")
                    or key.endswith("_objective_lens_center_z_mm")
                    or key == "upper_field_center_above_sample_mm"
                    or key == "virtual_lens_reference_z_mm"
                ):
                    payload.pop(key, None)
            for attribute in STRUCTURAL_FIELD_SOURCES:
                payload.pop(attribute, None)
            if str(component_key) == OBJECTIVE_LENS:
                for attribute in (
                    "assembly_outer_diameter_mm",
                    "pole_piece_center_separation_mm",
                    "pole_piece_axial_length_mm",
                    "upper_pole_piece_axial_length_mm",
                    "pole_piece_outer_diameter_mm",
                    "pole_piece_tip_diameter_mm",
                    "upper_pole_piece_outer_diameter_mm",
                    "upper_pole_piece_tip_diameter_mm",
                    "pole_piece_bore_diameter_mm",
                    "inner_face_gap_mm",
                    "virtual_lens_offset_below_lower_surface_mm",
                    "upper_b0_t",
                    "lower_b0_t",
                    "upper_a_mm",
                    "lower_a_mm",
                    "upper_gaussian",
                    "lower_gaussian",
                    "max_percent",
                    "nominal_voltage_kv",
                    "nominal_focal_length_mm",
                    "nominal_back_focal_plane_z_mm",
                    "nominal_image_plane_z_mm",
                    "cc_mm",
                ):
                    payload.pop(attribute, None)
            return payload

        def lens_payload(item):
            payload = asdict(item)
            if (
                canonical_component_placement_key(item.key)
                in TOML_OWNED_GEOMETRY_KEYS
            ):
                payload = strip_position_fields(payload, item.key)
            return payload

        def aperture_payload(item):
            payload = asdict(item)
            if (
                canonical_component_placement_key(item.key)
                in TOML_OWNED_GEOMETRY_KEYS
            ):
                payload = strip_position_fields(payload, item.key)
            return payload

        def component_payload(item):
            payload = asdict(item)
            if (
                canonical_component_placement_key(item.key)
                in TOML_OWNED_GEOMETRY_KEYS
            ):
                payload = strip_position_fields(payload, item.key)
            return payload

        if self.energy_filter is not None:
            ensure_energy_filter(self)
        corrector_mode = corrector_mode_for_hardware(
            self.corrector_mode,
            self.layout_c3_hardware,
        )
        correctors_allowed = (
            self.layout_c3_hardware != "two_condenser"
        )
        payload = {

            "schema_version": self.schema_version,

            "lenses":[lens_payload(x) for x in self.lenses],

            "apertures":[aperture_payload(x) for x in self.apertures],

            "stigmators":[component_payload(x) for x in self.stigmators],

            "deflectors":[component_payload(x) for x in self.deflectors],

            "electron_gun":self.electron_gun.to_dict(),
            "electron_gun_profiles":{
                str(key):dict(value)
                for key,value in self.electron_gun_profiles.items()
                if key != self.electron_gun.type_key
            },
            "sample":asdict(self.sample),

            "illumination_mode":self.illumination_mode,

            "component_placements":{
                str(key):dict(value)
                for key,value in self.component_placements.items()
                if key not in TOML_OWNED_GEOMETRY_KEYS
            },

            "projector_mode":self.projector_mode,
            "equivalent_image_lenses_enabled":(
                self.equivalent_image_lenses_enabled
            ),
            "step_mm":self.step_mm,

            "history_step_mm":self.history_step_mm,"acceleration_enabled":self.acceleration_enabled,"acceleration_backend":self.acceleration_backend,
            "active_backend":self.active_backend,
            "chromatic_aberration_enabled":self.chromatic_aberration_enabled,
            "corrector_mode":corrector_mode,
            "energy_filter_mode":self.energy_filter_mode,
            "c2c3_crossover_required":self.c2c3_crossover_required,

            "objective_coupled":self.objective_coupled,

            "recording_planes":serialise_recording_system(self),
            "corrector_elements":serialise_corrector_structure(self),

            "probe_corrector_installed":(
                correctors_allowed
                and getattr(self,"probe_corrector_installed",True)
            ),

            "image_corrector_installed":(
                correctors_allowed
                and getattr(self,"image_corrector_installed",False)
            ),

            "energy_filter_installed":getattr(self,"energy_filter_installed",True),
            "energy_filter":serialise_energy_filter(self.energy_filter),

            "column_mode":getattr(self,"column_mode","three_lens"),

            "layout_c3_hardware":getattr(self,"layout_c3_hardware","three_condenser"),

            "layout_c3_excited":getattr(self,"layout_c3_excited",True),
            "wobble_observation_plane_key":getattr(
                self, "wobble_observation_plane_key", "flu_screen"
            ),
            "wobble_custom_observation_z_mm":getattr(
                self, "wobble_custom_observation_z_mm", self.sample.z_mm
            ),
            "virtual_observation_z_mm":getattr(
                self, "virtual_observation_z_mm", self.sample.z_mm
            ),

            "layout_reference_positions":dict(
                getattr(self, "layout_reference_positions", {})
            ),
            "layout_reference_enabled":dict(
                getattr(self, "layout_reference_enabled", {})
            ),
            "monochromator_axis_offset_mm":float(
                self.monochromator_axis_offset_mm
            ),
            "ac_downstream_anchor_offsets_mm":{
                str(mode): {
                    str(key): float(value)
                    for key, value in offsets.items()
                }
                for mode, offsets in getattr(
                    self, "ac_downstream_anchor_offsets_mm", {}
                ).items()
            },
            "mini_condenser_upstream_gap_mm":float(
                self.mini_condenser_upstream_gap_mm
            ),
            "objective_stigmator_symmetry_offset_mm":float(
                self.objective_stigmator_symmetry_offset_mm
            ),
            "image_diffraction_deflector_upstream_gap_mm":float(
                self.image_diffraction_deflector_upstream_gap_mm
            ),
            "image_corrector_upstream_gap_mm":float(
                self.image_corrector_upstream_gap_mm
            ),
            "image_corrector_component_offsets_from_ol_post_mm":{
                str(key): float(value)
                for key, value in (
                    self.image_corrector_component_offsets_from_ol_post_mm
                ).items()
            },
            "selected_area_aperture_offset_from_sad_mm":float(
                self.selected_area_aperture_offset_from_sad_mm
            ),
            "standalone_selected_area_aperture_gap_after_descan_mm":float(
                self.standalone_selected_area_aperture_gap_after_descan_mm
            ),

        }
        for key in (
            "layout_reference_positions",
            "ac_downstream_anchor_offsets_mm",
            "mini_condenser_upstream_gap_mm",
            "objective_stigmator_symmetry_offset_mm",
            "image_diffraction_deflector_upstream_gap_mm",
            "image_corrector_upstream_gap_mm",
            "image_corrector_component_offsets_from_ol_post_mm",
            "selected_area_aperture_offset_from_sad_mm",
            "standalone_selected_area_aperture_gap_after_descan_mm",
        ):
            payload.pop(key, None)
        for collection in ("corrector_elements", "recording_planes"):
            payload[collection] = [
                (
                    strip_position_fields(item, item.get("key"))
                    if canonical_component_placement_key(
                        item.get("key", "")
                    ) in TOML_OWNED_GEOMETRY_KEYS
                    else item
                )
                for item in payload.get(collection, ())
            ]
        payload["sample"] = strip_position_fields(payload["sample"], "sample")
        return payload

    @staticmethod

    def from_dict(d):
        from temsim.configuration import (
            canonical_corrector_mode,
            corrector_mode_for_hardware,
        )
        loaded_schema_version = int(d.get("schema_version", 0))
        from temsim.component_keys import (
            AC_DEFLECTOR,
            ADAPTER_LENS,
            BEAM_DEFLECTOR,
            CONDENSER_APERTURE_2,
            CONDENSER_APERTURE_3,
            CONDENSER_DEFLECTOR,
            CONDENSER_LENS_1,
            CONDENSER_LENS_KEYS,
            CONDENSER_STIGMATOR,
            DC_DEFLECTOR,
            DESCAN_DEFLECTOR,
            DIFFRACTION_LENS,
            DIFFRACTION_STIGMATOR,
            ENERGY_FILTER_ENTRANCE_APERTURE,
            INTERMEDIATE_LENS,
            PROJECTOR_LENS_1,
            PROJECTOR_LENS_2,
            IMAGE_DIFFRACTION_DEFLECTOR,
            IMAGE_CORRECTOR_ELEMENT_KEYS,
            IMAGE_CORRECTOR_LENS_KEYS,
            IMAGE_CORRECTOR_OL_POST_LENS,
            MINI_CONDENSER,
            OBJECTIVE_APERTURE,
            OBJECTIVE_LENS,
            OBJECTIVE_STIGMATOR,
            SELECTED_AREA_APERTURE,
            PROBE_DPH2_DEFLECTOR,
            PROBE_DPH1_DEFLECTOR,
            PROBE_DP11_DEFLECTOR,
            PROBE_DP12_SCAN_DEFLECTOR,
            PROBE_DP21_DEFLECTOR,
            PROBE_DP22_DEFLECTOR,
            PROBE_HP1_HEXAPOLE,
            PROBE_HPOL_HEXAPOLE,
            PROBE_HP2_HEXAPOLE,
            PROBE_HPC_HEXAPOLE,
            PROBE_QPC_QUADRUPOLE,
            PROBE_QPH1_QUADRUPOLE,
            PROBE_QPOL_QUADRUPOLE,
            PROBE_QPH2_QUADRUPOLE,
            PROBE_TL12_LENS,
            PROBE_TL21_LENS,
            PROBE_TL22_LENS,
            canonical_aperture_key,
            canonical_binding_key,
            canonical_component_placement_key,
            canonical_corrector_element_key,
            canonical_deflector_key,
            canonical_lens_key,
            canonical_stigmator_key,
        )

        sample_data = dict(d.get("sample", {}))
        # Saved sample Z is legacy geometry. The selected assembly TOML is
        # authoritative; only specimen properties survive deserialisation.
        sample_z_mm = _DEFAULT_SAMPLE_Z_MM
        sample_thickness_nm = float(
            sample_data.get("thickness_nm", 100.0)
        )
        lens_rows = d.get("lenses", [])
        corrector_mode_for_load = canonical_corrector_mode(
            d.get("corrector_mode", "probe_corrector")
        )
        canonical_objective_present = any(
            str(item.get("key", "")) == OBJECTIVE_LENS
            for item in lens_rows
        )
        legacy_objective_mechanics = next(
            (
                item for item in lens_rows
                if str(item.get("key", "")) == OBJECTIVE_LENS
            ),
            {},
        )
        objective_loaded = False
        legacy_upper_objective = next(
            (
                item for item in lens_rows
                if str(item.get("key", "")) == "uobj"
            ),
            None,
        )
        legacy_lower_objective = next(
            (
                item for item in lens_rows
                if str(item.get("key", "")) == "lobj"
            ),
            None,
        )

        lenses=[]
        probe_lens_keys = {
            ADAPTER_LENS,
            MINI_CONDENSER,
            PROBE_TL22_LENS,
            PROBE_TL21_LENS,
            PROBE_TL12_LENS,
        }
        canonical_probe_lenses_present = {
            str(item.get("key", ""))
            for item in d["lenses"]
            if str(item.get("key", "")) in probe_lens_keys
        }
        loaded_probe_lens_keys = set()
        canonical_diffraction_lens_present = any(
            str(item.get("key", "")) == DIFFRACTION_LENS
            for item in lens_rows
        )
        diffraction_lens_loaded = False
        canonical_intermediate_lens_present = any(
            str(item.get("key", "")) == INTERMEDIATE_LENS
            for item in lens_rows
        )
        intermediate_lens_loaded = False
        canonical_projector_lens_p1_present = any(
            str(item.get("key", "")) == PROJECTOR_LENS_1
            for item in lens_rows
        )
        projector_lens_p1_loaded = False
        canonical_projector_lens_p2_present = any(
            str(item.get("key", "")) == PROJECTOR_LENS_2
            for item in lens_rows
        )
        projector_lens_p2_loaded = False
        anchor_row = next(
            (
                item
                for item in lens_rows
                if str(item.get("key", "")) == DIFFRACTION_LENS
            ),
            next(
                (
                    item
                    for item in lens_rows
                    if str(item.get("key", "")) == "diff"
                ),
                {},
            ),
        )
        legacy_anchor_reference = float(
            anchor_row.get(
                "standalone_optical_reference_z_mm",
                d.get("layout_reference_positions", {}).get(
                    f"lens:{DIFFRACTION_LENS}",
                    d.get("layout_reference_positions", {}).get(
                        "lens:diff",
                        anchor_row.get("z_mm", 1090.0),
                    ),
                ),
            )
        )

        for item in d["lenses"]:

            q=dict(item)
            raw_lens_key = str(q.get("key", ""))
            if raw_lens_key in {"uobj", "lobj", OBJECTIVE_LENS}:
                if objective_loaded:
                    continue
                if (
                    canonical_objective_present
                    and raw_lens_key != OBJECTIVE_LENS
                ):
                    continue
                if raw_lens_key == OBJECTIVE_LENS:
                    from temsim.optics.objective_lens import (
                        objective_lens_from_dict,
                    )
                    lenses.append(objective_lens_from_dict(
                        q, sample_z_mm, sample_thickness_nm
                    ))
                else:
                    from temsim.optics.objective_lens import (
                        objective_lens_from_legacy_rows,
                    )
                    lenses.append(objective_lens_from_legacy_rows(
                        legacy_upper_objective,
                        legacy_lower_objective,
                        sample_z_mm,
                        sample_thickness_nm,
                    ))
                objective_loaded = True
                continue
            q["key"]=canonical_lens_key(raw_lens_key)
            if q["key"] == DIFFRACTION_LENS:
                if (
                    diffraction_lens_loaded
                    or (
                        canonical_diffraction_lens_present
                        and raw_lens_key != DIFFRACTION_LENS
                    )
                ):
                    continue
                from temsim.optics.diffraction_lens import (
                    IMAGE_CORRECTED_INSTALLATION,
                    STANDALONE_INSTALLATION,
                    diffraction_lens_from_dict,
                )
                reference_positions = d.get(
                    "layout_reference_positions", {}
                )
                legacy_reference = reference_positions.get(
                    "lens:diff",
                    reference_positions.get(
                        f"lens:{DIFFRACTION_LENS}"
                    ),
                )
                lenses.append(diffraction_lens_from_dict(
                    q,
                    legacy_reference_z_mm=legacy_reference,
                    active_installation=(
                        IMAGE_CORRECTED_INSTALLATION
                        if corrector_mode_for_load in {
                            "image_corrector",
                            "double_corrector",
                        }
                        else STANDALONE_INSTALLATION
                    ),
                ))
                diffraction_lens_loaded = True
                continue
            if q["key"] == INTERMEDIATE_LENS:
                if (
                    intermediate_lens_loaded
                    or (
                        canonical_intermediate_lens_present
                        and raw_lens_key != INTERMEDIATE_LENS
                    )
                ):
                    continue
                from temsim.optics.intermediate_lens import (
                    intermediate_lens_from_dict,
                )
                reference_positions = d.get(
                    "layout_reference_positions", {}
                )
                lenses.append(intermediate_lens_from_dict(
                    q,
                    legacy_anchor_reference_z_mm=(
                        legacy_anchor_reference
                    ),
                    legacy_reference_z_mm=reference_positions.get(
                        f"lens:{INTERMEDIATE_LENS}",
                        reference_positions.get("lens:il"),
                    ),
                ))
                intermediate_lens_loaded = True
                continue
            if q["key"] == PROJECTOR_LENS_1:
                if (
                    projector_lens_p1_loaded
                    or (
                        canonical_projector_lens_p1_present
                        and raw_lens_key != PROJECTOR_LENS_1
                    )
                ):
                    continue
                from temsim.optics.projector_lens_p1 import (
                    projector_lens_p1_from_dict,
                )
                reference_positions = d.get(
                    "layout_reference_positions", {}
                )
                lenses.append(projector_lens_p1_from_dict(
                    q,
                    legacy_anchor_reference_z_mm=(
                        legacy_anchor_reference
                    ),
                    legacy_reference_z_mm=reference_positions.get(
                        f"lens:{PROJECTOR_LENS_1}",
                        reference_positions.get("lens:p1"),
                    ),
                ))
                projector_lens_p1_loaded = True
                continue
            if q["key"] == PROJECTOR_LENS_2:
                if (
                    projector_lens_p2_loaded
                    or (
                        canonical_projector_lens_p2_present
                        and raw_lens_key != PROJECTOR_LENS_2
                    )
                ):
                    continue
                from temsim.optics.projector_lens_p2 import (
                    projector_lens_p2_from_dict,
                )
                reference_positions = d.get(
                    "layout_reference_positions", {}
                )
                lenses.append(projector_lens_p2_from_dict(
                    q,
                    legacy_anchor_reference_z_mm=(
                        legacy_anchor_reference
                    ),
                    legacy_reference_z_mm=reference_positions.get(
                        f"lens:{PROJECTOR_LENS_2}",
                        reference_positions.get("lens:p2"),
                    ),
                ))
                projector_lens_p2_loaded = True
                continue
            if q["key"] in probe_lens_keys and (
                q["key"] in loaded_probe_lens_keys
                or (
                    q["key"] in canonical_probe_lenses_present
                    and raw_lens_key != q["key"]
                )
            ):
                continue
            if q["key"] in CONDENSER_LENS_KEYS:
                from temsim.optics.condenser_lens import (
                    condenser_lens_state_from_dict,
                )
                lenses.append(condenser_lens_state_from_dict(q))
            elif q["key"] in IMAGE_CORRECTOR_LENS_KEYS:
                from temsim.optics.image_corrector import (
                    image_corrector_component_from_dict,
                )
                lenses.append(
                    image_corrector_component_from_dict(q)
                )
            elif q["key"] == ADAPTER_LENS:
                from temsim.optics.probe_corrector import (
                    adapter_lens_from_dict,
                )
                lenses.append(adapter_lens_from_dict(q))
                loaded_probe_lens_keys.add(q["key"])
            elif q["key"] == MINI_CONDENSER:
                from temsim.optics.mini_condenser import (
                    mini_condenser_from_dict,
                )
                lenses.append(mini_condenser_from_dict(q))
                loaded_probe_lens_keys.add(q["key"])
            elif q["key"] == PROBE_TL22_LENS:
                from temsim.optics.probe_corrector import (
                    tl22_lens_from_dict,
                )
                lenses.append(tl22_lens_from_dict(q))
                loaded_probe_lens_keys.add(q["key"])
            elif q["key"] == PROBE_TL21_LENS:
                from temsim.optics.probe_corrector import (
                    tl21_lens_from_dict,
                )
                lenses.append(tl21_lens_from_dict(q))
                loaded_probe_lens_keys.add(q["key"])
            elif q["key"] == PROBE_TL12_LENS:
                from temsim.optics.probe_corrector import (
                    tl12_lens_from_dict,
                )
                lenses.append(tl12_lens_from_dict(q))
                loaded_probe_lens_keys.add(q["key"])
            else:
                q["gaussian"]=[
                    Gaussian(**g) for g in q.get("gaussian",[])
                ]
                lenses.append(Lens(**q))

        if not diffraction_lens_loaded:
            from temsim.optics.diffraction_lens import (
                create_diffraction_lens,
            )
            component = create_diffraction_lens()
            insertion_index = next(
                (
                    index
                    for index, lens in enumerate(lenses)
                    if lens.key == INTERMEDIATE_LENS
                ),
                len(lenses),
            )
            lenses.insert(insertion_index, component)
        if not intermediate_lens_loaded:
            from temsim.optics.intermediate_lens import (
                create_intermediate_lens,
            )
            component = create_intermediate_lens()
            insertion_index = next(
                (
                    index
                    for index, lens in enumerate(lenses)
                    if lens.key == PROJECTOR_LENS_1
                ),
                len(lenses),
            )
            lenses.insert(insertion_index, component)
        if not projector_lens_p1_loaded:
            from temsim.optics.projector_lens_p1 import (
                create_projector_lens_p1,
            )
            component = create_projector_lens_p1()
            insertion_index = next(
                (
                    index
                    for index, lens in enumerate(lenses)
                    if lens.key == PROJECTOR_LENS_2
                ),
                len(lenses),
            )
            lenses.insert(insertion_index, component)
        if not projector_lens_p2_loaded:
            from temsim.optics.projector_lens_p2 import (
                create_projector_lens_p2,
            )
            lenses.append(create_projector_lens_p2())

        apertures = []
        objective_aperture_loaded = False
        selected_area_aperture_loaded = False
        energy_filter_entrance_aperture_loaded = False
        for item in d.get("apertures", []):
            q = dict(item)
            q["key"] = canonical_aperture_key(q.get("key", ""))
            if q["key"] == OBJECTIVE_APERTURE:
                if objective_aperture_loaded:
                    continue
                from temsim.optics.objective_aperture import (
                    objective_aperture_from_dict,
                )
                apertures.append(objective_aperture_from_dict(
                    q,
                    sample_z_mm,
                    legacy_objective_mechanics,
                ))
                objective_aperture_loaded = True
            elif q["key"] == SELECTED_AREA_APERTURE:
                if selected_area_aperture_loaded:
                    continue
                from temsim.optics.selected_area_aperture import (
                    selected_area_aperture_from_dict,
                )
                apertures.append(selected_area_aperture_from_dict(q))
                selected_area_aperture_loaded = True
            elif q["key"] == ENERGY_FILTER_ENTRANCE_APERTURE:
                if energy_filter_entrance_aperture_loaded:
                    continue
                from temsim.optics.energy_filter_entrance_aperture import (
                    energy_filter_entrance_aperture_from_dict,
                )
                apertures.append(
                    energy_filter_entrance_aperture_from_dict(q)
                )
                energy_filter_entrance_aperture_loaded = True
            elif q["key"] in (
                CONDENSER_APERTURE_2,
                CONDENSER_APERTURE_3,
            ):
                from temsim.optics.condenser_aperture import (
                    condenser_aperture_from_dict,
                )
                apertures.append(condenser_aperture_from_dict(q))
            else:
                apertures.append(Aperture(**q))
        if not objective_aperture_loaded:
            from temsim.optics.objective_aperture import (
                create_objective_aperture,
            )
            apertures.append(create_objective_aperture(sample_z_mm))
        if not selected_area_aperture_loaded:
            from temsim.optics.selected_area_aperture import (
                create_selected_area_aperture,
            )
            apertures.append(create_selected_area_aperture())
        if not energy_filter_entrance_aperture_loaded:
            from temsim.optics.energy_filter_entrance_aperture import (
                create_energy_filter_entrance_aperture,
            )
            apertures.append(create_energy_filter_entrance_aperture())
        apertures.sort(key=lambda aperture: float(aperture.z_mm))

        # Schema V10 introduced explicit CETCOR image-corrector round lenses.
        # Older states receive the zero-excitation field bindings without
        # changing any saved legacy lens values.
        from temsim.optics.column import image_corrector_lenses
        present_lenses = {lens.key for lens in lenses}
        lenses.extend(
            lens
            for lens in image_corrector_lenses()
            if lens.key not in present_lenses
        )
        from temsim.optics.probe_corrector import (
            create_adapter_lens,
            create_tl12_lens,
            create_tl21_lens,
            create_tl22_lens,
        )
        probe_lens_factories = {
            ADAPTER_LENS: create_adapter_lens,
            PROBE_TL22_LENS: create_tl22_lens,
            PROBE_TL21_LENS: create_tl21_lens,
            PROBE_TL12_LENS: create_tl12_lens,
        }
        from temsim.optics.mini_condenser import create_mini_condenser
        probe_lens_factories[MINI_CONDENSER] = create_mini_condenser
        present_lenses = {lens.key for lens in lenses}
        for key, factory in probe_lens_factories.items():
            if key in present_lenses:
                continue
            component = factory()
            legacy_row = next(
                (
                    item
                    for item in d.get("corrector_elements", [])
                    if canonical_corrector_element_key(
                        item.get("key", "")
                    ) == key
                ),
                None,
            )
            if legacy_row is not None:
                component.optical_reference_from_tip_mm = float(
                    legacy_row.get("z_mm", component.z_mm)
                )
            lenses.append(component)
        if not objective_loaded:
            from temsim.optics.objective_lens import create_objective_lens
            objective = create_objective_lens(
                sample_z_mm, sample_thickness_nm
            )
            insertion_index = next(
                (
                    index
                    for index, lens in enumerate(lenses)
                    if float(lens.z_mm) > float(objective.z_mm)
                ),
                len(lenses),
            )
            lenses.insert(insertion_index, objective)

        loaded_deflectors = []
        canonical_dp12_present = any(
            str(item.get("key", "")) == PROBE_DP12_SCAN_DEFLECTOR
            for item in d.get("deflectors", [])
        )
        canonical_image_deflector_present = any(
            str(item.get("key", ""))
            == IMAGE_DIFFRACTION_DEFLECTOR
            for item in d.get("deflectors", [])
        )
        dp12_loaded = False
        image_deflector_loaded = False
        for item in d.get("deflectors", []):
            q = dict(item)
            raw_deflector_key = str(q.get("key", ""))
            # Very old files temporarily reused cond_def for the beam
            # shift/tilt pair. Resolve that ambiguity before canonicalisation.
            if q.get("key") == "cond_def" and "/ Beam" in q.get("name", ""):
                q["name"], q["key"] = (
                    "BSh/BTlt Beam Shift/Tilt Deflectors",
                    "beam_def",
                )
                if (
                    q.get("upper_z_mm") == 865
                    and q.get("lower_z_mm") == 905
                ):
                    q["upper_z_mm"], q["lower_z_mm"] = 900, 918
            q["key"] = canonical_deflector_key(q.get("key", ""))
            if q["key"] == PROBE_DP12_SCAN_DEFLECTOR and (
                dp12_loaded
                or (
                    canonical_dp12_present
                    and raw_deflector_key != PROBE_DP12_SCAN_DEFLECTOR
                )
            ):
                continue
            if q["key"] == CONDENSER_DEFLECTOR:
                from temsim.optics.condenser_deflector import (
                    condenser_deflector_from_dict,
                )
                loaded_deflectors.append(
                    condenser_deflector_from_dict(q)
                )
            elif q["key"] == BEAM_DEFLECTOR:
                from temsim.optics.beam_deflector import (
                    beam_deflector_from_dict,
                )
                loaded_deflectors.append(
                    beam_deflector_from_dict(q)
                )
            elif q["key"] == PROBE_DP12_SCAN_DEFLECTOR:
                from temsim.optics.probe_corrector import (
                    dp12_scan_deflector_from_dict,
                )
                loaded_deflectors.append(
                    dp12_scan_deflector_from_dict(q)
                )
                dp12_loaded = True
            elif q["key"] == IMAGE_DIFFRACTION_DEFLECTOR:
                if (
                    image_deflector_loaded
                    or (
                        canonical_image_deflector_present
                        and raw_deflector_key
                        != IMAGE_DIFFRACTION_DEFLECTOR
                    )
                ):
                    continue
                from temsim.optics.image_diffraction_deflector import (
                    image_diffraction_deflector_from_dict,
                )
                loaded_deflectors.append(
                    image_diffraction_deflector_from_dict(q)
                )
                image_deflector_loaded = True
            else:
                known = DeflectorPair.__dataclass_fields__
                loaded_deflectors.append(DeflectorPair(**{
                    key: value
                    for key, value in q.items()
                    if key in known
                }))

        for item in loaded_deflectors:
            if (
                item.key == IMAGE_DIFFRACTION_DEFLECTOR
                and item.upper_z_mm == 1215
                and item.lower_z_mm == 1265
            ):

                item.upper_z_mm,item.lower_z_mm=975,1000
                if hasattr(item, "optical_upper_reference_z_mm"):
                    item.optical_upper_reference_z_mm = 975
                    item.optical_lower_reference_z_mm = 1000

        present={item.key for item in loaded_deflectors}

        if CONDENSER_DEFLECTOR not in present:
            from temsim.optics.condenser_deflector import (
                create_condenser_deflector,
            )
            loaded_deflectors.append(create_condenser_deflector())

        if BEAM_DEFLECTOR not in present:
            from temsim.optics.beam_deflector import create_beam_deflector
            loaded_deflectors.append(create_beam_deflector())

        if PROBE_DP12_SCAN_DEFLECTOR not in present:
            from temsim.optics.probe_corrector import (
                create_dp12_scan_deflector,
            )
            loaded_deflectors.append(create_dp12_scan_deflector())

        if IMAGE_DIFFRACTION_DEFLECTOR not in present:
            from temsim.optics.image_diffraction_deflector import (
                create_image_diffraction_deflector,
            )
            loaded_deflectors.append(
                create_image_diffraction_deflector()
            )

        # Preserve the physical column order, including newer devices such as
        # post-scan coils that were not part of the legacy migration map.
        loaded_deflectors.sort(key=lambda item:(item.upper_z_mm+item.lower_z_mm)/2.0)

        gun_data = d.get("electron_gun")
        if not isinstance(gun_data, dict):
            raise ValueError(
                "State data must contain one canonical electron_gun object."
            )
        from temsim.optics.electron_gun import create_electron_gun
        electron_gun = create_electron_gun(
            gun_data.get("type", ""), gun_data
        )
        if (
            electron_gun.type_key == "cold_feg"
            and "monochromator" not in gun_data
            and bool(d.get("monochromator_installed", False))
        ):
            electron_gun.migrate_legacy_monochromator_bay()
        from temsim.column.module_assembly import TOML_OWNED_GEOMETRY_KEYS
        component_placements={
            canonical_component_placement_key(key):dict(value)
            for key,value in d.get("component_placements",{}).items()
            if (
                canonical_component_placement_key(key)
                not in TOML_OWNED_GEOMETRY_KEYS
            )
        }
        stigmator_rows = d.get("stigmators", [])
        canonical_stigmator_present = any(
            str(item.get("key", "")) == CONDENSER_STIGMATOR
            for item in stigmator_rows
        )
        canonical_diffraction_stigmator_present = any(
            str(item.get("key", "")) == DIFFRACTION_STIGMATOR
            for item in stigmator_rows
        )
        loaded_stigmators = []
        condenser_stigmator_loaded = False
        objective_stigmator_loaded = False
        diffraction_stigmator_loaded = False
        for item in stigmator_rows:
            values = dict(item)
            raw_key = str(values.get("key", ""))
            if raw_key == "hpol":
                continue
            values["key"] = canonical_stigmator_key(raw_key)
            if values["key"] == CONDENSER_STIGMATOR:
                if (
                    condenser_stigmator_loaded
                    or (
                        canonical_stigmator_present
                        and raw_key != CONDENSER_STIGMATOR
                    )
                ):
                    continue
                from temsim.optics.condenser_stigmator import (
                    condenser_stigmator_from_dict,
                )
                loaded_stigmators.append(
                    condenser_stigmator_from_dict(values)
                )
                condenser_stigmator_loaded = True
            elif values["key"] == OBJECTIVE_STIGMATOR:
                if objective_stigmator_loaded:
                    continue
                from temsim.optics.objective_stigmator import (
                    objective_stigmator_from_dict,
                )
                loaded_stigmators.append(
                    objective_stigmator_from_dict(values)
                )
                objective_stigmator_loaded = True
            elif values["key"] == DIFFRACTION_STIGMATOR:
                if (
                    diffraction_stigmator_loaded
                    or (
                        canonical_diffraction_stigmator_present
                        and raw_key != DIFFRACTION_STIGMATOR
                    )
                ):
                    continue
                from temsim.optics.diffraction_stigmator import (
                    IMAGE_CORRECTED_INSTALLATION,
                    STANDALONE_INSTALLATION,
                    diffraction_stigmator_from_dict,
                )
                corrector_mode = canonical_corrector_mode(
                    d.get("corrector_mode", "probe_corrector")
                )
                legacy_reference = d.get(
                    "layout_reference_positions", {}
                ).get(
                    "stigmator:diff_stig",
                    d.get("layout_reference_positions", {}).get(
                        f"stigmator:{DIFFRACTION_STIGMATOR}"
                    ),
                )
                loaded_stigmators.append(
                    diffraction_stigmator_from_dict(
                        values,
                        legacy_reference_z_mm=legacy_reference,
                        active_installation=(
                            IMAGE_CORRECTED_INSTALLATION
                            if corrector_mode in {
                                "image_corrector",
                                "double_corrector",
                            }
                            else STANDALONE_INSTALLATION
                        ),
                    )
                )
                diffraction_stigmator_loaded = True
            else:
                loaded_stigmators.append(Stigmator(**values))
        if not condenser_stigmator_loaded:
            from temsim.optics.condenser_stigmator import (
                create_condenser_stigmator,
            )
            loaded_stigmators.append(create_condenser_stigmator())
        if not objective_stigmator_loaded:
            from temsim.optics.objective_stigmator import (
                create_objective_stigmator,
            )
            loaded_stigmators.append(create_objective_stigmator())
        if not diffraction_stigmator_loaded:
            from temsim.optics.diffraction_stigmator import (
                create_diffraction_stigmator,
            )
            loaded_stigmators.append(create_diffraction_stigmator())
        loaded_stigmators.sort(key=lambda item: float(item.z_mm))
        from temsim import module_manifest
        manifest_monochromator_offset_mm = (
            module_manifest.port_z_mm("gun/FEG_Mono.toml", "exit")
            - module_manifest.port_z_mm("gun/FEG.toml", "exit")
            if electron_gun.type_key == "cold_feg"
            and electron_gun.monochromator_installed
            else 0.0
        )

        state=State(
            lenses=lenses, apertures=apertures,
            stigmators=loaded_stigmators,
            deflectors=loaded_deflectors,
            electron_gun=electron_gun,
            electron_gun_profiles={
                str(key):dict(value)
                for key,value in d.get(
                    "electron_gun_profiles", {}
                ).items()
                if str(key) != electron_gun.type_key
            },
            sample=Sample(**sample_data), camera=None,
            component_placements=component_placements,
            illumination_mode=d.get("illumination_mode","STEM"), projector_mode=d.get("projector_mode","diffraction"),
            equivalent_image_lenses_enabled=bool(
                d.get("equivalent_image_lenses_enabled", False)
            ),
            step_mm=d.get("step_mm",0.5),
            history_step_mm=d.get("history_step_mm",2.0), acceleration_enabled=d.get("acceleration_enabled",True),
            acceleration_backend=d.get("acceleration_backend","Auto"), active_backend=d.get("active_backend","CPU"),
            corrector_mode=corrector_mode_for_hardware(
                d.get("corrector_mode", "probe_corrector"),
                d.get("layout_c3_hardware", "three_condenser"),
            ), energy_filter_mode=d.get("energy_filter_mode","energy_filter"),
            column_mode=d.get("column_mode","three_lens"), c2c3_crossover_required=d.get("c2c3_crossover_required",True),
            objective_coupled=d.get("objective_coupled",True),
            corrector_crossover_targets_mm=[],
            layout_c3_hardware=d.get("layout_c3_hardware","three_condenser"),
            layout_c3_excited=d.get("layout_c3_excited",d.get("column_mode","three_lens")=="three_lens"),
            layout_reference_positions={
                canonical_binding_key(key): float(value)
                for key, value in d.get("layout_reference_positions", {}).items()
            },
            layout_reference_enabled={
                canonical_binding_key(key): bool(value)
                for key, value in d.get("layout_reference_enabled", {}).items()
            },
            monochromator_column_offset_mm=float(
                d.get(
                    "monochromator_column_offset_mm",
                    manifest_monochromator_offset_mm,
                )
            ),
            monochromator_axis_offset_mm=float(
                d.get("monochromator_axis_offset_mm", 0.0)
            ),
            ac_downstream_anchor_offsets_mm={
                str(mode): {
                    str(key): float(value)
                    for key, value in offsets.items()
                }
                for mode, offsets in d.get(
                    "ac_downstream_anchor_offsets_mm", {}
                ).items()
                if isinstance(offsets, dict)
            },
            mini_condenser_upstream_gap_mm=float(
                d.get("mini_condenser_upstream_gap_mm", 0.0)
            ),
            objective_stigmator_symmetry_offset_mm=float(
                d.get("objective_stigmator_symmetry_offset_mm", 0.0)
            ),
            image_diffraction_deflector_upstream_gap_mm=float(
                d.get(
                    "image_diffraction_deflector_upstream_gap_mm",
                    0.0,
                )
            ),
            image_corrector_upstream_gap_mm=float(
                d.get("image_corrector_upstream_gap_mm", 5.0)
            ),
            image_corrector_component_offsets_from_ol_post_mm={
                str(key): float(value)
                for key, value in d.get(
                    "image_corrector_component_offsets_from_ol_post_mm",
                    {},
                ).items()
            },
            selected_area_aperture_offset_from_sad_mm=float(
                d.get(
                    "selected_area_aperture_offset_from_sad_mm",
                    0.0,
                )
            ),
            standalone_selected_area_aperture_gap_after_descan_mm=float(
                d.get(
                    "standalone_selected_area_aperture_gap_after_descan_mm",
                    5.0,
                )
            ),
            wobble_observation_plane_key=d.get(
                "wobble_observation_plane_key", "flu_screen"
            ),
            wobble_custom_observation_z_mm=float(
                d.get(
                    "wobble_custom_observation_z_mm",
                    _DEFAULT_SAMPLE_Z_MM,
                )
            ),
            virtual_observation_z_mm=float(
                d.get(
                    "virtual_observation_z_mm",
                    d.get(
                        "wobble_custom_observation_z_mm",
                        _DEFAULT_SAMPLE_Z_MM,
                    ),
                )
            ),
            chromatic_aberration_enabled=bool(
                d.get("chromatic_aberration_enabled", False)
            ),
            schema_version=64,
        )
        if loaded_schema_version < 64:
            from temsim.specimen.geometry import (
                quaternion_from_euler_xyz_deg,
                set_sample_orientation,
            )
            from temsim.specimen.virtual import (
                legacy_virtual_interaction_rows,
            )

            set_sample_orientation(
                state.sample,
                quaternion_from_euler_xyz_deg(
                    (
                        state.sample.specimen_rotation_x_deg,
                        state.sample.specimen_rotation_y_deg,
                        state.sample.specimen_rotation_z_deg,
                    )
                ),
            )
            state.sample.virtual_interactions = (
                legacy_virtual_interaction_rows(state.sample)
            )
        state.probe_corrector_installed=d.get("probe_corrector_installed",True)
        state.image_corrector_installed=d.get("image_corrector_installed",False)
        if state.layout_c3_hardware == "two_condenser":
            state.corrector_mode = "no_corrector"
            state.probe_corrector_installed = False
            state.image_corrector_installed = False
        state.energy_filter_installed=d.get("energy_filter_installed",True)
        filter_data=d.get("energy_filter")
        if filter_data is not None:
            from temsim.optics.energy_filter import energy_filter_from_dict
            state.energy_filter=energy_filter_from_dict(
                filter_data,
                state.beam_voltage_kv,
            )
        from temsim.detector.recording_system import restore_recording_system
        from temsim.optics.corrector_structure import CorrectorElement, ensure_corrector_structure
        elements=d.get("corrector_elements")
        if elements:
            allowed=CorrectorElement.__dataclass_fields__
            state.corrector_elements = []
            modular_keys = {
                PROBE_DPH2_DEFLECTOR,
                PROBE_DPH1_DEFLECTOR,
                PROBE_DP11_DEFLECTOR,
                PROBE_DP21_DEFLECTOR,
                PROBE_DP22_DEFLECTOR,
                PROBE_HP1_HEXAPOLE,
                PROBE_HPOL_HEXAPOLE,
                PROBE_QPH2_QUADRUPOLE,
                PROBE_QPC_QUADRUPOLE,
                PROBE_QPH1_QUADRUPOLE,
                PROBE_QPOL_QUADRUPOLE,
                PROBE_HP2_HEXAPOLE,
                PROBE_HPC_HEXAPOLE,
                AC_DEFLECTOR,
                DESCAN_DEFLECTOR,
                *IMAGE_CORRECTOR_ELEMENT_KEYS,
            }
            canonical_rows = {
                str(item.get("key", ""))
                for item in elements
                if str(item.get("key", "")) in modular_keys
            }
            seen_modular_keys = set()
            for item in elements:
                values = dict(item)
                raw_key = str(values.get("key", ""))
                if raw_key in {
                    "ic_hpol_qpol_dp11",
                    "ic_dsh_dstg",
                    "ic_ol_post",
                    "ic_tl11",
                    "ic_tl21",
                    "ic_tl22",
                    "ic_adl",
                }:
                    continue
                values["key"] = canonical_corrector_element_key(
                    raw_key
                )
                if values["key"] == DC_DEFLECTOR:
                    # Legacy projects may contain the removed DC deflector.
                    # It is deliberately discarded during migration.
                    continue
                if (
                    values["key"] in canonical_rows
                    and raw_key != values["key"]
                ):
                    continue
                if values["key"] in modular_keys:
                    if values["key"] in seen_modular_keys:
                        continue
                    seen_modular_keys.add(values["key"])
                if values["key"] in {
                    PROBE_TL22_LENS,
                    PROBE_TL21_LENS,
                    PROBE_TL12_LENS,
                    PROBE_DP12_SCAN_DEFLECTOR,
                }:
                    continue
                if values["key"] == PROBE_DPH2_DEFLECTOR:
                    from temsim.optics.probe_corrector import (
                        dph2_deflector_from_dict,
                    )
                    state.corrector_elements.append(
                        dph2_deflector_from_dict(values)
                    )
                elif values["key"] == PROBE_DP22_DEFLECTOR:
                    from temsim.optics.probe_corrector import (
                        dp22_deflector_from_dict,
                    )
                    state.corrector_elements.append(
                        dp22_deflector_from_dict(values)
                    )
                elif values["key"] == PROBE_QPH2_QUADRUPOLE:
                    from temsim.optics.probe_corrector import (
                        qph2_quadrupole_from_dict,
                    )
                    state.corrector_elements.append(
                        qph2_quadrupole_from_dict(values)
                    )
                elif values["key"] == PROBE_HP2_HEXAPOLE:
                    from temsim.optics.probe_corrector import (
                        hp2_hexapole_from_dict,
                    )
                    state.corrector_elements.append(
                        hp2_hexapole_from_dict(values)
                    )
                elif values["key"] == PROBE_HPC_HEXAPOLE:
                    from temsim.optics.probe_corrector import (
                        hpc_hexapole_from_dict,
                    )
                    state.corrector_elements.append(
                        hpc_hexapole_from_dict(values)
                    )
                elif values["key"] == PROBE_QPC_QUADRUPOLE:
                    from temsim.optics.probe_corrector import (
                        qpc_quadrupole_from_dict,
                    )
                    state.corrector_elements.append(
                        qpc_quadrupole_from_dict(values)
                    )
                elif values["key"] == PROBE_DP21_DEFLECTOR:
                    from temsim.optics.probe_corrector import (
                        dp21_deflector_from_dict,
                    )
                    state.corrector_elements.append(
                        dp21_deflector_from_dict(values)
                    )
                elif values["key"] == PROBE_DPH1_DEFLECTOR:
                    from temsim.optics.probe_corrector import (
                        dph1_deflector_from_dict,
                    )
                    state.corrector_elements.append(
                        dph1_deflector_from_dict(values)
                    )
                elif values["key"] == PROBE_QPH1_QUADRUPOLE:
                    from temsim.optics.probe_corrector import (
                        qph1_quadrupole_from_dict,
                    )
                    state.corrector_elements.append(
                        qph1_quadrupole_from_dict(values)
                    )
                elif values["key"] == PROBE_HP1_HEXAPOLE:
                    from temsim.optics.probe_corrector import (
                        hp1_hexapole_from_dict,
                    )
                    state.corrector_elements.append(
                        hp1_hexapole_from_dict(values)
                    )
                elif values["key"] == PROBE_HPOL_HEXAPOLE:
                    from temsim.optics.probe_corrector import (
                        hpol_hexapole_from_dict,
                    )
                    state.corrector_elements.append(
                        hpol_hexapole_from_dict(values)
                    )
                elif values["key"] == PROBE_QPOL_QUADRUPOLE:
                    from temsim.optics.probe_corrector import (
                        qpol_quadrupole_from_dict,
                    )
                    state.corrector_elements.append(
                        qpol_quadrupole_from_dict(values)
                    )
                elif values["key"] == PROBE_DP11_DEFLECTOR:
                    from temsim.optics.probe_corrector import (
                        dp11_deflector_from_dict,
                    )
                    state.corrector_elements.append(
                        dp11_deflector_from_dict(values)
                    )
                elif values["key"] == AC_DEFLECTOR:
                    from temsim.optics.ac_deflector import (
                        ac_deflector_from_dict,
                    )
                    state.corrector_elements.append(
                        ac_deflector_from_dict(values)
                    )
                elif values["key"] == DESCAN_DEFLECTOR:
                    from temsim.optics.descan_deflector import (
                        descan_deflector_from_dict,
                    )
                    state.corrector_elements.append(
                        descan_deflector_from_dict(values)
                    )
                elif values["key"] in IMAGE_CORRECTOR_ELEMENT_KEYS:
                    from temsim.optics.image_corrector import (
                        image_corrector_component_from_dict,
                    )
                    state.corrector_elements.append(
                        image_corrector_component_from_dict(values)
                    )
                else:
                    state.corrector_elements.append(
                        CorrectorElement(**{
                            key: value
                            for key, value in values.items()
                            if key in allowed
                        })
                    )
        restore_recording_system(state,d.get("recording_planes"))
        if filter_data is not None:
            from temsim.optics.energy_filter import ensure_energy_filter
            ensure_energy_filter(state)
        if (
            not energy_filter_entrance_aperture_loaded
            and isinstance(filter_data, dict)
        ):
            entrance = state.energy_filter_entrance_aperture
            if "entrance_aperture_mm" in filter_data:
                entrance.radius_mm = min(
                    max(
                        float(filter_data["entrance_aperture_mm"])
                        / 2.0,
                        0.0,
                    ),
                    float(entrance.maximum_radius_mm),
                )
            legacy_z_mm = filter_data.get("entrance_z_mm")
            if (
                legacy_z_mm is not None
                and abs(float(legacy_z_mm) - 2200.0) > 1.0e-9
            ):
                entrance.set_optical_reference_z_mm(
                    state.selected_area_aperture.z_mm,
                    float(legacy_z_mm),
                )
            entrance.validate()
        ensure_corrector_structure(state)
        if loaded_schema_version < 52:
            from temsim.optics.beam_deflector import (
                resolve_beam_deflector_after_active_aperture,
            )
            from temsim.optics.probe_corrector import (
                anchor_probe_corrector_to_beam_deflector,
                synchronise_probe_corrector_physical_axis,
            )
            resolve_beam_deflector_after_active_aperture(state)
            anchor_probe_corrector_to_beam_deflector(state)
            synchronise_probe_corrector_physical_axis(state)
        if loaded_schema_version < 53:
            state.objective_stigmator.mechanical_length_mm = 30.0
            state.layout_reference_positions.pop(
                "stigmator:objective_stigmator", None
            )
        if loaded_schema_version < 54:
            state.mini_condenser.mechanical_length_mm = 30.0
            state.mini_condenser.standalone_mechanical_length_mm = 30.0
            for offsets in state.ac_downstream_anchor_offsets_mm.values():
                if isinstance(offsets, dict):
                    offsets.pop("mini_condenser", None)
        if loaded_schema_version < 55:
            state.mini_condenser_upstream_gap_mm = 0.0
            for offsets in state.ac_downstream_anchor_offsets_mm.values():
                if isinstance(offsets, dict):
                    offsets.pop("mini_condenser", None)
        if loaded_schema_version < 56:
            image_deflector = state.image_diffraction_deflector
            image_deflector.thickness_mm = 15.0
            image_deflector.inter_coil_gap_mm = 10.0
            image_deflector.mechanical_length_mm = 40.0
            image_deflector.optical_plane_separation_mm = 25.0
            state.image_diffraction_deflector_upstream_gap_mm = 0.0
            state.layout_reference_positions.pop(
                "deflector:image_diffraction_deflector", None
            )
            for offsets in state.ac_downstream_anchor_offsets_mm.values():
                if isinstance(offsets, dict):
                    offsets.pop("descan_deflector", None)
            if selected_area_aperture_loaded:
                selected_area_aperture = state.selected_area_aperture
                (
                    selected_area_aperture
                    .standalone_mechanical_center_below_sample_mm
                ) += 30.0
                (
                    selected_area_aperture
                    .image_corrected_mechanical_center_below_sample_mm
                ) += 30.0
            from temsim.component_keys import (
                IMAGE_CORRECTOR_ADAPTER_LENS,
                IMAGE_CORRECTOR_DP21_DEFLECTOR,
                IMAGE_CORRECTOR_DP22_DEFLECTOR,
                IMAGE_CORRECTOR_DSH_DEFLECTOR,
                IMAGE_CORRECTOR_DSTG_QUADRUPOLE,
                IMAGE_CORRECTOR_HP1_HEXAPOLE,
                IMAGE_CORRECTOR_HP2_HEXAPOLE,
                IMAGE_CORRECTOR_ISH_DEFLECTOR,
                IMAGE_CORRECTOR_SAD_PLANE,
                IMAGE_CORRECTOR_TL21_LENS,
                IMAGE_CORRECTOR_TL22_LENS,
            )
            downstream_image_corrector_keys = {
                IMAGE_CORRECTOR_HP1_HEXAPOLE,
                IMAGE_CORRECTOR_DP21_DEFLECTOR,
                IMAGE_CORRECTOR_TL21_LENS,
                IMAGE_CORRECTOR_DP22_DEFLECTOR,
                IMAGE_CORRECTOR_TL22_LENS,
                IMAGE_CORRECTOR_HP2_HEXAPOLE,
                IMAGE_CORRECTOR_ADAPTER_LENS,
                IMAGE_CORRECTOR_ISH_DEFLECTOR,
                IMAGE_CORRECTOR_DSH_DEFLECTOR,
                IMAGE_CORRECTOR_DSTG_QUADRUPOLE,
                IMAGE_CORRECTOR_SAD_PLANE,
            }
            serialized_image_corrector_keys = {
                str(item.get("key", ""))
                for collection_name in ("lenses", "corrector_elements")
                for item in d.get(collection_name, [])
            }
            for component in state.image_corrector_system.components:
                if (
                    component.key in downstream_image_corrector_keys
                    and component.key in serialized_image_corrector_keys
                ):
                    state._translate_component_from_tip(component, 30.0)
        if loaded_schema_version < 57:
            from temsim.component_keys import (
                IMAGE_CORRECTOR_ADAPTER_LENS,
                IMAGE_CORRECTOR_DP21_DEFLECTOR,
                IMAGE_CORRECTOR_DP22_DEFLECTOR,
                IMAGE_CORRECTOR_DSH_DEFLECTOR,
                IMAGE_CORRECTOR_DSTG_QUADRUPOLE,
                IMAGE_CORRECTOR_HP1_HEXAPOLE,
                IMAGE_CORRECTOR_HP2_HEXAPOLE,
                IMAGE_CORRECTOR_ISH_DEFLECTOR,
                IMAGE_CORRECTOR_SAD_PLANE,
                IMAGE_CORRECTOR_TL21_LENS,
                IMAGE_CORRECTOR_TL22_LENS,
            )
            downstream_image_corrector_keys = {
                IMAGE_CORRECTOR_HP1_HEXAPOLE,
                IMAGE_CORRECTOR_DP21_DEFLECTOR,
                IMAGE_CORRECTOR_TL21_LENS,
                IMAGE_CORRECTOR_DP22_DEFLECTOR,
                IMAGE_CORRECTOR_TL22_LENS,
                IMAGE_CORRECTOR_HP2_HEXAPOLE,
                IMAGE_CORRECTOR_ADAPTER_LENS,
                IMAGE_CORRECTOR_ISH_DEFLECTOR,
                IMAGE_CORRECTOR_DSH_DEFLECTOR,
                IMAGE_CORRECTOR_DSTG_QUADRUPOLE,
                IMAGE_CORRECTOR_SAD_PLANE,
            }
            serialized_image_corrector_keys = {
                str(item.get("key", ""))
                for collection_name in ("lenses", "corrector_elements")
                for item in d.get(collection_name, [])
            }
            state.objective_stigmator_symmetry_offset_mm = 0.0
            for offsets in state.ac_downstream_anchor_offsets_mm.values():
                if isinstance(offsets, dict):
                    offsets.pop("objective_stigmator", None)
                    offsets.pop("descan_deflector", None)
            if selected_area_aperture_loaded:
                selected_area_aperture = state.selected_area_aperture
                (
                    selected_area_aperture
                    .standalone_mechanical_center_below_sample_mm
                ) -= 90.0
                (
                    selected_area_aperture
                    .image_corrected_mechanical_center_below_sample_mm
                ) -= 90.0
            for component in state.image_corrector_system.components:
                if (
                    component.key in downstream_image_corrector_keys
                    and component.key in serialized_image_corrector_keys
                ):
                    state._translate_component_from_tip(component, -90.0)
        if loaded_schema_version < 58:
            from temsim.optics.image_corrector import (
                DEFAULT_IMAGE_CORRECTOR_UPSTREAM_GAP_MM,
                DEFAULT_SELECTED_AREA_APERTURE_OFFSET_FROM_SAD_MM,
            )
            image_corrector_components = (
                state.image_corrector_system.components
            )
            ol_post_center_mm = float(
                state.image_corrector_system.ol_post_lens
                .mechanical_center_from_tip_mm
            )
            state.image_corrector_upstream_gap_mm = (
                DEFAULT_IMAGE_CORRECTOR_UPSTREAM_GAP_MM
            )
            state.image_corrector_component_offsets_from_ol_post_mm = {
                component.key: (
                    float(component.mechanical_center_from_tip_mm)
                    - ol_post_center_mm
                )
                for component in image_corrector_components
                if component.key != IMAGE_CORRECTOR_OL_POST_LENS
            }
            state.selected_area_aperture_offset_from_sad_mm = (
                DEFAULT_SELECTED_AREA_APERTURE_OFFSET_FROM_SAD_MM
            )
            for component in image_corrector_components:
                prefix = (
                    "lens"
                    if component.key in IMAGE_CORRECTOR_LENS_KEYS
                    else "corrector"
                )
                state.layout_reference_positions.pop(
                    f"{prefix}:{component.key}", None
                )
            state.layout_reference_positions.pop(
                f"aperture:{SELECTED_AREA_APERTURE}", None
            )
        if loaded_schema_version < 59:
            from temsim.optics.selected_area_aperture import (
                DEFAULT_STANDALONE_GAP_AFTER_DESCAN_MM,
            )
            state.standalone_selected_area_aperture_gap_after_descan_mm = (
                DEFAULT_STANDALONE_GAP_AFTER_DESCAN_MM
            )
            state.layout_reference_positions.pop(
                f"aperture:{SELECTED_AREA_APERTURE}", None
            )
        if loaded_schema_version < 60:
            downstream_defaults = {
                "probe": {
                    "condenser_stigmator": 120.0,
                    "objective_upper_lens": 145.0,
                    "objective_upper_pole": 145.0,
                    "sample": 155.0,
                    "objective_aperture": 156.5,
                    "objective_lower_pole": 165.0,
                    "objective_lower_lens": 165.0,
                    "descan_deflector": 265.0,
                },
                "standalone": {
                    "condenser_stigmator": 175.0,
                    "objective_upper_lens": 200.0,
                    "objective_upper_pole": 200.0,
                    "sample": 210.0,
                    "objective_aperture": 211.5,
                    "objective_lower_pole": 220.0,
                    "objective_lower_lens": 220.0,
                    "descan_deflector": 320.0,
                },
            }
            for mode, defaults in downstream_defaults.items():
                offsets = (
                    state.ac_downstream_anchor_offsets_mm
                    .setdefault(mode, {})
                )
                if not isinstance(offsets, dict):
                    offsets = {}
                    state.ac_downstream_anchor_offsets_mm[mode] = offsets
                for key, value in defaults.items():
                    offsets.setdefault(key, value)
                current_stigmator_offset_mm = float(
                    offsets["condenser_stigmator"]
                )
                target_stigmator_offset_mm = (
                    70.0 + float(state.mini_condenser_upstream_gap_mm)
                )
                rigid_shift_mm = (
                    target_stigmator_offset_mm
                    - current_stigmator_offset_mm
                )
                for key in defaults:
                    offsets[key] = (
                        float(offsets[key]) + rigid_shift_mm
                    )
        if loaded_schema_version < 62:
            # V62 installs the current two-hexapole corrector calibration.
            # Strengths change, but all manifest-owned component centres stay
            # fixed.  ``None`` Cs values deliberately select the conventional
            # positive magnetic-lens estimate in the propagation core.
            from temsim.optics.probe_corrector import (
                PROBE_HP1_HEXAPOLE_ORIENTATION_RAD,
                PROBE_HP1_HEXAPOLE_STRENGTH_RATIO,
                PROBE_MAIN_HEXAPOLE_STRENGTH_M3,
            )
            probe = state.probe_corrector_system
            probe.tl22_lens.b0_t = 0.31809425
            probe.tl22_lens.percent = 100.0
            probe.tl21_lens.b0_t = 0.29864759
            probe.tl21_lens.percent = 100.0
            probe.hp2_hexapole.strength_m3 = (
                PROBE_MAIN_HEXAPOLE_STRENGTH_M3
            )
            probe.hp2_hexapole.orientation_rad = 0.0
            probe.hp1_hexapole.strength_m3 = (
                PROBE_MAIN_HEXAPOLE_STRENGTH_M3
                * PROBE_HP1_HEXAPOLE_STRENGTH_RATIO
            )
            probe.hp1_hexapole.orientation_rad = (
                PROBE_HP1_HEXAPOLE_ORIENTATION_RAD
            )

            from temsim.optics.image_corrector import (
                IMAGE_HP2_HEXAPOLE_ORIENTATION_RAD,
                IMAGE_HP2_HEXAPOLE_STRENGTH_RATIO,
                IMAGE_MAIN_HEXAPOLE_STRENGTH_M3,
            )
            image = state.image_corrector_system
            image_lens_fields = (
                (image.ol_post_lens, 1.82167110),
                (image.tl11_lens, 0.33677618),
                (image.tl12_lens, 0.19797227),
                (image.tl21_lens, 1.29298746),
                (image.tl22_lens, 1.27157328),
                (image.adapter_lens, 0.25184796),
            )
            for lens, field_t in image_lens_fields:
                lens.b0_t = field_t
                lens.percent = 100.0
            image.hp1_hexapole.strength_m3 = (
                IMAGE_MAIN_HEXAPOLE_STRENGTH_M3
            )
            image.hp1_hexapole.orientation_rad = 0.0
            image.hp2_hexapole.strength_m3 = (
                IMAGE_MAIN_HEXAPOLE_STRENGTH_M3
                * IMAGE_HP2_HEXAPOLE_STRENGTH_RATIO
            )
            image.hp2_hexapole.orientation_rad = (
                IMAGE_HP2_HEXAPOLE_ORIENTATION_RAD
            )
        if loaded_schema_version < 63:
            # V63 adds rated-field headroom to every lens whose calibrated
            # default had reached 100%.  Scale maximum field and excitation
            # inversely so existing projects keep exactly the same Bz.
            c2 = state.condenser_lens_2.lens
            c2.b0_t /= 0.7
            c2.percent *= 0.7

            probe = state.probe_corrector_system
            for lens in (
                probe.tl22_lens,
                probe.tl21_lens,
                probe.tl12_lens,
            ):
                lens.b0_t /= 0.6
                lens.percent *= 0.6

            for lens in state.image_corrector_system.round_lens_components:
                lens.b0_t /= 0.6
                lens.percent *= 0.6

            # Objective field ratings are manifest-owned, so from_dict has
            # already loaded the larger V63 rating; only the saved percentage
            # needs rebasing to preserve the previous physical field.
            state.objective_lens.percent *= 0.7
        from temsim.optics.beam_deflector import (
            resolve_beam_deflector_after_active_aperture,
        )
        from temsim.optics.probe_corrector import (
            anchor_probe_corrector_to_beam_deflector,
            synchronise_probe_corrector_physical_axis,
        )
        # Position records are no longer authoritative for this rigid chain,
        # even in a current-schema file. Rebuild it from its one root.
        resolve_beam_deflector_after_active_aperture(state)
        anchor_probe_corrector_to_beam_deflector(state)
        synchronise_probe_corrector_physical_axis(state)
        if state.monochromator_installed:
            from temsim import module_manifest
            state._set_monochromator_column_offset(
                module_manifest.port_z_mm(
                    "gun/FEG_Mono.toml", "exit"
                )
                - module_manifest.port_z_mm(
                    "gun/FEG.toml", "exit"
                )
            )
        else:
            state._set_monochromator_column_offset(0.0)
        from temsim.component_names import normalise_component_names
        normalise_component_names(state)
        state.sync_objective()
        from temsim.column.state_layout import apply_physical_layout_to_state
        apply_physical_layout_to_state(state)
        return state
