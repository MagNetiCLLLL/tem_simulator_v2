"""Position coupling only; physical fields and dataclass schemas stay in components."""

STANDALONE_INSTALLATION = "standalone"
IMAGE_CORRECTED_INSTALLATION = "image_corrected"


class TipReferencedPosition:
    """Move the mechanical centre and preserve its optical offset from the tip.

    No dataclass fields: construction/restoration can disable coupling.
    A direct z edit remains temporary until apply_optical_position().
    """

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        if name in {
            "z_mm",
            "mechanical_center_from_tip_mm",
            "optical_reference_from_tip_mm",
        }:
            value = float(value)
        coupling_ready = self.__dict__.get(
            "_position_coupling_ready", False
        )
        if name == "mechanical_center_from_tip_mm" and coupling_ready:
            delta_mm = float(value) - float(
                self.mechanical_center_from_tip_mm
            )
            object.__setattr__(self, name, float(value))
            optical = float(self.optical_reference_from_tip_mm) + delta_mm
            object.__setattr__(
                self, "optical_reference_from_tip_mm", optical
            )
            object.__setattr__(self, "z_mm", optical)
            return
        if name == "optical_reference_from_tip_mm" and coupling_ready:
            object.__setattr__(self, name, float(value))
            object.__setattr__(self, "z_mm", float(value))
            return
        object.__setattr__(self, name, value)

    def apply_optical_position(self):
        self.z_mm = float(self.optical_reference_from_tip_mm)
        return self


class InstallationReferencedPosition:
    """Couple only the active installation while retaining the inactive geometry."""

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        ready = self.__dict__.get("_position_coupling_ready", False)
        center_attributes = {
            "standalone_mechanical_center_below_sample_mm": (
                STANDALONE_INSTALLATION
            ),
            "image_corrected_mechanical_center_below_sample_mm": (
                IMAGE_CORRECTED_INSTALLATION
            ),
        }
        reference_attributes = {
            "standalone_optical_reference_z_mm": STANDALONE_INSTALLATION,
            "image_corrected_optical_reference_z_mm": (
                IMAGE_CORRECTED_INSTALLATION
            ),
        }
        if ready and name in center_attributes:
            value = float(value)
            delta_mm = value - float(getattr(self, name))
            installation = center_attributes[name]
            reference_attribute = (
                f"{installation}_optical_reference_z_mm"
            )
            reference = float(getattr(self, reference_attribute)) + delta_mm
            object.__setattr__(self, name, value)
            object.__setattr__(self, reference_attribute, reference)
            if self.active_installation == installation:
                object.__setattr__(self, "z_mm", reference)
            return
        if ready and name in reference_attributes:
            value = float(value)
            object.__setattr__(self, name, value)
            if self.active_installation == reference_attributes[name]:
                object.__setattr__(self, "z_mm", value)
            return
        if ready and name == "z_mm":
            value = float(value)
            object.__setattr__(self, name, value)
            object.__setattr__(
                self,
                f"{self.active_installation}_optical_reference_z_mm",
                value,
            )
            return
        object.__setattr__(self, name, value)

    def select_installation(self, installation):
        geometry = self.geometry_for(installation)
        object.__setattr__(self, "active_installation", installation)
        object.__setattr__(
            self, "z_mm", float(geometry.optical_reference_z_mm)
        )
        return self

    def resolve_against(self, selected_area_geometry):
        mechanical_center = (
            float(selected_area_geometry.mechanical_center_below_sample_mm)
            + float(self.mechanical_center_downstream_of_anchor_mm)
        )
        optical_reference = (
            float(selected_area_geometry.optical_reference_z_mm)
            + float(self.mechanical_center_downstream_of_anchor_mm)
        )
        object.__setattr__(
            self,
            "optical_reference_downstream_of_anchor_mm",
            float(self.mechanical_center_downstream_of_anchor_mm),
        )
        object.__setattr__(
            self,
            f"{self.active_installation}"
            "_mechanical_center_below_sample_mm",
            mechanical_center,
        )
        object.__setattr__(
            self,
            f"{self.active_installation}_optical_reference_z_mm",
            optical_reference,
        )
        object.__setattr__(self, "z_mm", optical_reference)
        return self.geometry_for(self.active_installation)
