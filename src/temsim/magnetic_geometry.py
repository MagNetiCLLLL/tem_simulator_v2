"""Shared physical material intervals, independent of display envelopes."""


def objective_layer_intervals_mm(parent_data, parent_start_mm, profile):
    fields = ("local_start_z_mm", "upper_yoke_start_local_z_mm", "upper_yoke_end_local_z_mm",
              "lower_yoke_start_local_z_mm", "lower_yoke_end_local_z_mm")
    if any(key not in parent_data for key in fields):
        return ()
    origin = float(parent_start_mm)-float(parent_data["local_start_z_mm"])
    inset = float(parent_data.get("mechanical_coil_axial_inset_mm", 0)) if profile == "magnetic_excitation_coil" else 0
    intervals = tuple((origin+float(parent_data[f"{side}_yoke_start_local_z_mm"])+inset,
                       origin+float(parent_data[f"{side}_yoke_end_local_z_mm"])-inset)
                      for side in ("upper", "lower"))
    if any(end <= start for start, end in intervals):
        raise ValueError("Objective layer intervals must have positive length")
    return intervals
