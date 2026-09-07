# Project dimension audit

11 module documents · 482 component definitions · 4480 numeric dimension values · 280 review items.

Scope: saved documents only. Repeated components in alternative modules are counted separately. Runtime openings and unsaved drafts are excluded. A physical meaning does not establish a measured source; declared provenance is reported without independent certification. Full values and exact array paths are in the JSON report.

## Classification

| Meaning | Values |
| --- | ---: |
| envelope | 538 |
| operating | 168 |
| physical | 1070 |
| placement | 1742 |
| unknown | 436 |
| vacuum | 526 |

| Source | Values |
| --- | ---: |
| estimated | 182 |
| unspecified | 4298 |

## Review items

| Issue | Count |
| --- | ---: |
| envelope_only_geometry | 183 |
| undefined_outer_envelope | 5 |
| undefined_strip_outline | 23 |
| undefined_working_opening | 23 |
| unimplemented_geometry_parameter | 16 |
| zero_thickness_reference | 30 |

- **envelope_only_geometry** — `E:\tem_simulator_v2\configs\instruments\beam_blanker\NanoPulser.toml`: This mechanism/profile has an envelope preview; its internal or non-axisymmetric construction is not a fully specified CAD solid.
  Affected: `parts.nanopulser_deflector.mechanical_profile`.

- **envelope_only_geometry** — `E:\tem_simulator_v2\configs\instruments\column\C2.toml`: This mechanism/profile has an envelope preview; its internal or non-axisymmetric construction is not a fully specified CAD solid.
  Affected: `parts.condenser_lens_1.mechanical_profile`; `parts.objective_lens.mechanical_profile`; `parts.condenser_stigmator.mechanical_profile`; `parts.beam_deflector.mechanical_profile`; `parts.ac_deflector.mechanical_profile`; `parts.mini_condenser.mechanical_profile`; `parts.sample.mechanical_profile`; `parts.descan_deflector.mechanical_profile`; `parts.objective_stigmator.mechanical_profile`; `parts.image_diffraction_deflector.mechanical_profile`; `parts.condenser_lens_2.mechanical_profile`; `parts.c1_c2_pole_piece_cartridge.mechanical_profile`.

- **undefined_outer_envelope** — `E:\tem_simulator_v2\configs\instruments\column\C2.toml`: No supported outer radius/diameter/width is declared, so the existing preview cannot establish an outer body envelope.
  Affected: `parts.sample_stage.mechanical_outer_diameter_mm`.

- **undefined_strip_outline** — `E:\tem_simulator_v2\configs\instruments\column\C2.toml`: Transverse strip dimensions and hole positions are not specified; the outline is schematic, not measured CAD.
  Affected: `parts.condenser_aperture_2.aperture_plate_form`; `parts.objective_aperture.aperture_plate_form`.

- **undefined_working_opening** — `E:\tem_simulator_v2\configs\instruments\column\C2.toml`: The saved file does not define a working opening. A carrier bore or maximum operating radius is not the current opening; runtime state may provide it.
  Affected: `parts.condenser_aperture_2.radius_mm`; `parts.objective_aperture.radius_mm`.

- **zero_thickness_reference** — `E:\tem_simulator_v2\configs\instruments\column\C2.toml`: This component is a zero-thickness reference/envelope plane. The preview does not invent a physical plate thickness.
  Affected: `parts.eds_detector_system.length_mm`.

- **envelope_only_geometry** — `E:\tem_simulator_v2\configs\instruments\column\C3.toml`: This mechanism/profile has an envelope preview; its internal or non-axisymmetric construction is not a fully specified CAD solid.
  Affected: `parts.condenser_lens_1.mechanical_profile`; `parts.objective_lens.mechanical_profile`; `parts.condenser_lens_2.mechanical_profile`; `parts.condenser_stigmator.mechanical_profile`; `parts.beam_deflector.mechanical_profile`; `parts.ac_deflector.mechanical_profile`; `parts.mini_condenser.mechanical_profile`; `parts.sample.mechanical_profile`; `parts.descan_deflector.mechanical_profile`; `parts.objective_stigmator.mechanical_profile`; `parts.image_diffraction_deflector.mechanical_profile`; `parts.condenser_deflector.mechanical_profile`; `parts.condenser_lens_3.mechanical_profile`; `parts.c1_c2_pole_piece_cartridge.mechanical_profile`.

- **undefined_outer_envelope** — `E:\tem_simulator_v2\configs\instruments\column\C3.toml`: No supported outer radius/diameter/width is declared, so the existing preview cannot establish an outer body envelope.
  Affected: `parts.sample_stage.mechanical_outer_diameter_mm`.

- **undefined_strip_outline** — `E:\tem_simulator_v2\configs\instruments\column\C3.toml`: Transverse strip dimensions and hole positions are not specified; the outline is schematic, not measured CAD.
  Affected: `parts.condenser_aperture_2.aperture_plate_form`; `parts.condenser_aperture_3.aperture_plate_form`; `parts.objective_aperture.aperture_plate_form`.

- **undefined_working_opening** — `E:\tem_simulator_v2\configs\instruments\column\C3.toml`: The saved file does not define a working opening. A carrier bore or maximum operating radius is not the current opening; runtime state may provide it.
  Affected: `parts.condenser_aperture_2.radius_mm`; `parts.condenser_aperture_3.radius_mm`; `parts.objective_aperture.radius_mm`.

- **zero_thickness_reference** — `E:\tem_simulator_v2\configs\instruments\column\C3.toml`: This component is a zero-thickness reference/envelope plane. The preview does not invent a physical plate thickness.
  Affected: `parts.eds_detector_system.length_mm`.

- **envelope_only_geometry** — `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml`: This mechanism/profile has an envelope preview; its internal or non-axisymmetric construction is not a fully specified CAD solid.
  Affected: `parts.condenser_lens_1.mechanical_profile`; `parts.objective_lens.mechanical_profile`; `parts.condenser_lens_2.mechanical_profile`; `parts.condenser_stigmator.mechanical_profile`; `parts.beam_deflector.mechanical_profile`; `parts.ac_deflector.mechanical_profile`; `parts.mini_condenser.mechanical_profile`; `parts.sample.mechanical_profile`; `parts.descan_deflector.mechanical_profile`; `parts.objective_stigmator.mechanical_profile`; `parts.image_diffraction_deflector.mechanical_profile`; `parts.condenser_deflector.mechanical_profile`; `parts.condenser_lens_3.mechanical_profile`; `parts.image_ol_post_lens.mechanical_profile`; `parts.image_hpol_hexapole.mechanical_profile`; `parts.image_qpol_quadrupole.mechanical_profile`; `parts.image_dp11_deflector.mechanical_profile`; `parts.image_tl11_lens.mechanical_profile`; `parts.image_dp12_deflector.mechanical_profile`; `parts.image_tl12_lens.mechanical_profile`; `parts.image_dph1_deflector.mechanical_profile`; `parts.image_hp1_hexapole.mechanical_profile`; `parts.image_dp21_deflector.mechanical_profile`; `parts.image_tl21_lens.mechanical_profile`; `parts.image_dp22_deflector.mechanical_profile`; `parts.image_tl22_lens.mechanical_profile`; `parts.image_dph2_deflector.mechanical_profile`; `parts.image_hp2_hexapole.mechanical_profile`; `parts.image_adapter_lens.mechanical_profile`; `parts.image_ish_deflector.mechanical_profile`; `parts.image_dsh_deflector.mechanical_profile`; `parts.image_dstg_quadrupole.mechanical_profile`; `parts.c1_c2_pole_piece_cartridge.mechanical_profile`.

- **undefined_outer_envelope** — `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml`: No supported outer radius/diameter/width is declared, so the existing preview cannot establish an outer body envelope.
  Affected: `parts.sample_stage.mechanical_outer_diameter_mm`.

- **undefined_strip_outline** — `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml`: Transverse strip dimensions and hole positions are not specified; the outline is schematic, not measured CAD.
  Affected: `parts.condenser_aperture_2.aperture_plate_form`; `parts.condenser_aperture_3.aperture_plate_form`; `parts.objective_aperture.aperture_plate_form`.

- **undefined_working_opening** — `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml`: The saved file does not define a working opening. A carrier bore or maximum operating radius is not the current opening; runtime state may provide it.
  Affected: `parts.condenser_aperture_2.radius_mm`; `parts.condenser_aperture_3.radius_mm`; `parts.objective_aperture.radius_mm`.

- **zero_thickness_reference** — `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml`: This component is a zero-thickness reference/envelope plane. The preview does not invent a physical plate thickness.
  Affected: `parts.image_sad_plane.length_mm`; `parts.eds_detector_system.length_mm`.

- **envelope_only_geometry** — `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml`: This mechanism/profile has an envelope preview; its internal or non-axisymmetric construction is not a fully specified CAD solid.
  Affected: `parts.condenser_lens_1.mechanical_profile`; `parts.condenser_deflector.mechanical_profile`; `parts.adapter_lens.mechanical_profile`; `parts.probe_dph2_deflector.mechanical_profile`; `parts.probe_qph2_quadrupole.mechanical_profile`; `parts.condenser_lens_2.mechanical_profile`; `parts.probe_hp2_hexapole.mechanical_profile`; `parts.probe_tl22_lens.mechanical_profile`; `parts.probe_dp22_deflector.mechanical_profile`; `parts.probe_hpc_hexapole.mechanical_profile`; `parts.probe_qpc_quadrupole.mechanical_profile`; `parts.probe_dp21_deflector.mechanical_profile`; `parts.probe_tl21_lens.mechanical_profile`; `parts.probe_dph1_deflector.mechanical_profile`; `parts.probe_qph1_quadrupole.mechanical_profile`; `parts.probe_hp1_hexapole.mechanical_profile`; `parts.probe_hpol_hexapole.mechanical_profile`; `parts.probe_qpol_quadrupole.mechanical_profile`; `parts.probe_dp11_deflector.mechanical_profile`; `parts.probe_tl12_lens.mechanical_profile`; `parts.objective_lens.mechanical_profile`; `parts.condenser_stigmator.mechanical_profile`; `parts.beam_deflector.mechanical_profile`; `parts.ac_deflector.mechanical_profile`; `parts.mini_condenser.mechanical_profile`; `parts.condenser_lens_3.mechanical_profile`; `parts.sample.mechanical_profile`; `parts.descan_deflector.mechanical_profile`; `parts.objective_stigmator.mechanical_profile`; `parts.image_diffraction_deflector.mechanical_profile`; `parts.c1_c2_pole_piece_cartridge.mechanical_profile`.

- **undefined_outer_envelope** — `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml`: No supported outer radius/diameter/width is declared, so the existing preview cannot establish an outer body envelope.
  Affected: `parts.sample_stage.mechanical_outer_diameter_mm`.

- **undefined_strip_outline** — `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml`: Transverse strip dimensions and hole positions are not specified; the outline is schematic, not measured CAD.
  Affected: `parts.condenser_aperture_2.aperture_plate_form`; `parts.condenser_aperture_3.aperture_plate_form`; `parts.objective_aperture.aperture_plate_form`.

- **undefined_working_opening** — `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml`: The saved file does not define a working opening. A carrier bore or maximum operating radius is not the current opening; runtime state may provide it.
  Affected: `parts.condenser_aperture_2.radius_mm`; `parts.condenser_aperture_3.radius_mm`; `parts.objective_aperture.radius_mm`.

- **zero_thickness_reference** — `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml`: This component is a zero-thickness reference/envelope plane. The preview does not invent a physical plate thickness.
  Affected: `parts.probe_dp12_scan_deflector.length_mm`; `parts.eds_detector_system.length_mm`.

- **envelope_only_geometry** — `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml`: This mechanism/profile has an envelope preview; its internal or non-axisymmetric construction is not a fully specified CAD solid.
  Affected: `parts.condenser_lens_1.mechanical_profile`; `parts.condenser_deflector.mechanical_profile`; `parts.adapter_lens.mechanical_profile`; `parts.probe_dph2_deflector.mechanical_profile`; `parts.probe_qph2_quadrupole.mechanical_profile`; `parts.condenser_lens_2.mechanical_profile`; `parts.probe_hp2_hexapole.mechanical_profile`; `parts.probe_tl22_lens.mechanical_profile`; `parts.probe_dp22_deflector.mechanical_profile`; `parts.probe_hpc_hexapole.mechanical_profile`; `parts.probe_qpc_quadrupole.mechanical_profile`; `parts.probe_dp21_deflector.mechanical_profile`; `parts.probe_tl21_lens.mechanical_profile`; `parts.probe_dph1_deflector.mechanical_profile`; `parts.probe_qph1_quadrupole.mechanical_profile`; `parts.probe_hp1_hexapole.mechanical_profile`; `parts.probe_hpol_hexapole.mechanical_profile`; `parts.probe_qpol_quadrupole.mechanical_profile`; `parts.probe_dp11_deflector.mechanical_profile`; `parts.probe_tl12_lens.mechanical_profile`; `parts.objective_lens.mechanical_profile`; `parts.condenser_stigmator.mechanical_profile`; `parts.beam_deflector.mechanical_profile`; `parts.ac_deflector.mechanical_profile`; `parts.mini_condenser.mechanical_profile`; `parts.condenser_lens_3.mechanical_profile`; `parts.sample.mechanical_profile`; `parts.descan_deflector.mechanical_profile`; `parts.objective_stigmator.mechanical_profile`; `parts.image_diffraction_deflector.mechanical_profile`; `parts.image_ol_post_lens.mechanical_profile`; `parts.image_hpol_hexapole.mechanical_profile`; `parts.image_qpol_quadrupole.mechanical_profile`; `parts.image_dp11_deflector.mechanical_profile`; `parts.image_tl11_lens.mechanical_profile`; `parts.image_dp12_deflector.mechanical_profile`; `parts.image_tl12_lens.mechanical_profile`; `parts.image_dph1_deflector.mechanical_profile`; `parts.image_hp1_hexapole.mechanical_profile`; `parts.image_dp21_deflector.mechanical_profile`; `parts.image_tl21_lens.mechanical_profile`; `parts.image_dp22_deflector.mechanical_profile`; `parts.image_tl22_lens.mechanical_profile`; `parts.image_dph2_deflector.mechanical_profile`; `parts.image_hp2_hexapole.mechanical_profile`; `parts.image_adapter_lens.mechanical_profile`; `parts.image_ish_deflector.mechanical_profile`; `parts.image_dsh_deflector.mechanical_profile`; `parts.image_dstg_quadrupole.mechanical_profile`; `parts.c1_c2_pole_piece_cartridge.mechanical_profile`.

- **undefined_outer_envelope** — `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml`: No supported outer radius/diameter/width is declared, so the existing preview cannot establish an outer body envelope.
  Affected: `parts.sample_stage.mechanical_outer_diameter_mm`.

- **undefined_strip_outline** — `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml`: Transverse strip dimensions and hole positions are not specified; the outline is schematic, not measured CAD.
  Affected: `parts.condenser_aperture_2.aperture_plate_form`; `parts.condenser_aperture_3.aperture_plate_form`; `parts.objective_aperture.aperture_plate_form`.

- **undefined_working_opening** — `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml`: The saved file does not define a working opening. A carrier bore or maximum operating radius is not the current opening; runtime state may provide it.
  Affected: `parts.condenser_aperture_2.radius_mm`; `parts.condenser_aperture_3.radius_mm`; `parts.objective_aperture.radius_mm`.

- **zero_thickness_reference** — `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml`: This component is a zero-thickness reference/envelope plane. The preview does not invent a physical plate thickness.
  Affected: `parts.probe_dp12_scan_deflector.length_mm`; `parts.image_sad_plane.length_mm`; `parts.eds_detector_system.length_mm`.

- **envelope_only_geometry** — `E:\tem_simulator_v2\configs\instruments\gun\FEG.toml`: This mechanism/profile has an envelope preview; its internal or non-axisymmetric construction is not a fully specified CAD solid.
  Affected: `parts.feg_tip.mechanical_profile`; `parts.feg_extractor.mechanical_profile`; `parts.feg_electrostatic_lens.mechanical_profile`; `parts.feg_accelerator.mechanical_profile`; `parts.feg_deflector.mechanical_profile`; `parts.feg_stigmator.mechanical_profile`.

- **undefined_strip_outline** — `E:\tem_simulator_v2\configs\instruments\gun\FEG.toml`: Transverse strip dimensions and hole positions are not specified; the outline is schematic, not measured CAD.
  Affected: `parts.feg_dpa_aperture.aperture_plate_form`; `parts.feg_c1_aperture.aperture_plate_form`.

- **undefined_working_opening** — `E:\tem_simulator_v2\configs\instruments\gun\FEG.toml`: The saved file does not define a working opening. A carrier bore or maximum operating radius is not the current opening; runtime state may provide it.
  Affected: `parts.feg_dpa_aperture.radius_mm`; `parts.feg_c1_aperture.radius_mm`.

- **envelope_only_geometry** — `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml`: This mechanism/profile has an envelope preview; its internal or non-axisymmetric construction is not a fully specified CAD solid.
  Affected: `parts.feg_tip.mechanical_profile`; `parts.feg_extractor.mechanical_profile`; `parts.feg_electrostatic_lens.mechanical_profile`; `parts.feg_monochromator_wien.mechanical_profile`; `parts.feg_accelerator.mechanical_profile`; `parts.feg_deflector.mechanical_profile`; `parts.feg_stigmator.mechanical_profile`; `parts.feg_monochromator_slit.mechanical_profile`.

- **undefined_strip_outline** — `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml`: Transverse strip dimensions and hole positions are not specified; the outline is schematic, not measured CAD.
  Affected: `parts.feg_dpa_aperture.aperture_plate_form`; `parts.feg_c1_aperture.aperture_plate_form`.

- **undefined_working_opening** — `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml`: The saved file does not define a working opening. A carrier bore or maximum operating radius is not the current opening; runtime state may provide it.
  Affected: `parts.feg_dpa_aperture.radius_mm`; `parts.feg_c1_aperture.radius_mm`.

- **envelope_only_geometry** — `E:\tem_simulator_v2\configs\instruments\gun\Thermionic.toml`: This mechanism/profile has an envelope preview; its internal or non-axisymmetric construction is not a fully specified CAD solid.
  Affected: `parts.thermionic_cathode.mechanical_profile`; `parts.thermionic_wehnelt.mechanical_profile`; `parts.thermionic_gun_lens.mechanical_profile`; `parts.thermionic_accelerator.mechanical_profile`; `parts.thermionic_deflector.mechanical_profile`; `parts.thermionic_stigmator.mechanical_profile`.

- **undefined_strip_outline** — `E:\tem_simulator_v2\configs\instruments\gun\Thermionic.toml`: Transverse strip dimensions and hole positions are not specified; the outline is schematic, not measured CAD.
  Affected: `parts.thermionic_anode_aperture.aperture_plate_form`; `parts.thermionic_c1_aperture.aperture_plate_form`.

- **undefined_working_opening** — `E:\tem_simulator_v2\configs\instruments\gun\Thermionic.toml`: The saved file does not define a working opening. A carrier bore or maximum operating radius is not the current opening; runtime state may provide it.
  Affected: `parts.thermionic_anode_aperture.radius_mm`; `parts.thermionic_c1_aperture.radius_mm`.

- **envelope_only_geometry** — `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml`: This mechanism/profile has an envelope preview; its internal or non-axisymmetric construction is not a fully specified CAD solid.
  Affected: `parts.diffraction_stigmator.mechanical_profile`; `parts.diffraction_lens.mechanical_profile`; `parts.intermediate_lens.mechanical_profile`; `parts.projector_lens_1.mechanical_profile`; `parts.projector_lens_2.mechanical_profile`; `parts.flu_screen.mechanical_profile`; `parts.haadf.mechanical_profile`; `parts.camera.mechanical_profile`; `parts.df.mechanical_profile`; `parts.bf.mechanical_profile`; `parts.post_projector_detector_chamber.mechanical_profile`.

- **undefined_strip_outline** — `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml`: Transverse strip dimensions and hole positions are not specified; the outline is schematic, not measured CAD.
  Affected: `parts.selected_area_aperture.aperture_plate_form`; `parts.energy_filter_entrance_aperture.aperture_plate_form`.

- **undefined_working_opening** — `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml`: The saved file does not define a working opening. A carrier bore or maximum operating radius is not the current opening; runtime state may provide it.
  Affected: `parts.selected_area_aperture.radius_mm`; `parts.energy_filter_entrance_aperture.radius_mm`.

- **unimplemented_geometry_parameter** — `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml`: The fillet range is reference metadata; the current solid retains sharp shoulders and does not construct a radius from this range.
  Affected: `parts.diffraction_lens_upper_pole.pole_root_fillet_radius_range_mm`; `parts.diffraction_lens_lower_pole.pole_root_fillet_radius_range_mm`; `parts.intermediate_lens_upper_pole.pole_root_fillet_radius_range_mm`; `parts.intermediate_lens_lower_pole.pole_root_fillet_radius_range_mm`; `parts.projector_lens_1_upper_pole.pole_root_fillet_radius_range_mm`; `parts.projector_lens_1_lower_pole.pole_root_fillet_radius_range_mm`; `parts.projector_lens_2_upper_pole.pole_root_fillet_radius_range_mm`; `parts.projector_lens_2_lower_pole.pole_root_fillet_radius_range_mm`.

- **zero_thickness_reference** — `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml`: This component is a zero-thickness reference/envelope plane. The preview does not invent a physical plate thickness.
  Affected: `parts.energy_filter.length_mm`; `parts.energy_filter_tapered_prism.length_mm`; `parts.energy_filter_multipole_01.length_mm`; `parts.energy_filter_multipole_02.length_mm`; `parts.energy_filter_multipole_03.length_mm`; `parts.energy_filter_multipole_04.length_mm`; `parts.energy_filter_multipole_05.length_mm`; `parts.energy_filter_multipole_06.length_mm`; `parts.energy_filter_multipole_07.length_mm`; `parts.energy_filter_multipole_08.length_mm`; `parts.energy_filter_multipole_09.length_mm`; `parts.energy_filter_multipole_10.length_mm`; `parts.energy_filter_slit.length_mm`; `parts.energy_filter_dynamic_focus_electrostatic_quadrupole.length_mm`; `parts.energy_filter_bias_tube.length_mm`; `parts.energy_filter_shutter.length_mm`; `parts.energy_filter_camera_deflector.length_mm`; `parts.energy_filter_eftem_output_plane.length_mm`; `parts.energy_filter_zebra.length_mm`; `parts.projection_chamber_dpa_aperture.length_mm`.

- **envelope_only_geometry** — `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml`: This mechanism/profile has an envelope preview; its internal or non-axisymmetric construction is not a fully specified CAD solid.
  Affected: `parts.diffraction_stigmator.mechanical_profile`; `parts.diffraction_lens.mechanical_profile`; `parts.intermediate_lens.mechanical_profile`; `parts.projector_lens_1.mechanical_profile`; `parts.projector_lens_2.mechanical_profile`; `parts.flu_screen.mechanical_profile`; `parts.haadf.mechanical_profile`; `parts.camera.mechanical_profile`; `parts.df.mechanical_profile`; `parts.bf.mechanical_profile`; `parts.post_projector_detector_chamber.mechanical_profile`.

- **undefined_strip_outline** — `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml`: Transverse strip dimensions and hole positions are not specified; the outline is schematic, not measured CAD.
  Affected: `parts.selected_area_aperture.aperture_plate_form`.

- **undefined_working_opening** — `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml`: The saved file does not define a working opening. A carrier bore or maximum operating radius is not the current opening; runtime state may provide it.
  Affected: `parts.selected_area_aperture.radius_mm`.

- **unimplemented_geometry_parameter** — `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml`: The fillet range is reference metadata; the current solid retains sharp shoulders and does not construct a radius from this range.
  Affected: `parts.diffraction_lens_upper_pole.pole_root_fillet_radius_range_mm`; `parts.diffraction_lens_lower_pole.pole_root_fillet_radius_range_mm`; `parts.intermediate_lens_upper_pole.pole_root_fillet_radius_range_mm`; `parts.intermediate_lens_lower_pole.pole_root_fillet_radius_range_mm`; `parts.projector_lens_1_upper_pole.pole_root_fillet_radius_range_mm`; `parts.projector_lens_1_lower_pole.pole_root_fillet_radius_range_mm`; `parts.projector_lens_2_upper_pole.pole_root_fillet_radius_range_mm`; `parts.projector_lens_2_lower_pole.pole_root_fillet_radius_range_mm`.

- **zero_thickness_reference** — `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml`: This component is a zero-thickness reference/envelope plane. The preview does not invent a physical plate thickness.
  Affected: `parts.projection_chamber_dpa_aperture.length_mm`.

## Evidence gaps by component

The following grouped counts identify values with unspecified evidence or an unclassified meaning. They are review needs, not a claim that the configured numbers are incorrect.

| File | Component | Unspecified source | Unknown meaning |
| --- | --- | ---: | ---: |
| `E:\tem_simulator_v2\configs\instruments\beam_blanker\NanoPulser.toml` | `(module)` | 7 | 2 |
| `E:\tem_simulator_v2\configs\instruments\beam_blanker\NanoPulser.toml` | `nanopulser_aperture` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\beam_blanker\NanoPulser.toml` | `nanopulser_deflector` | 10 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `(module)` | 16 | 7 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `ac_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `beam_deflector` | 12 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `c1_c2_pole_piece_cartridge` | 7 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `condenser_aperture_2` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `condenser_lens_1` | 12 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `condenser_lens_1_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `condenser_lens_1_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `condenser_lens_1_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `condenser_lens_1_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `condenser_lens_2` | 11 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `condenser_lens_2_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `condenser_lens_2_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `condenser_lens_2_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `condenser_lens_2_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `condenser_stigmator` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `descan_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `eds_detector_system` | 5 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `image_diffraction_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `mini_condenser` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `mini_condenser_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `mini_condenser_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `mini_condenser_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `mini_condenser_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `mini_condenser_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `objective_aperture` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `objective_lens` | 23 | 8 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `objective_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `objective_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `objective_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `objective_lower_pole` | 15 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `objective_stigmator` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `objective_upper_pole` | 15 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `sample` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C2.toml` | `sample_stage` | 7 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `(module)` | 16 | 7 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `ac_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `beam_deflector` | 12 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `c1_c2_pole_piece_cartridge` | 7 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_aperture_2` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_aperture_3` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_deflector` | 12 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_1` | 12 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_1_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_1_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_1_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_1_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_2` | 11 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_2_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_2_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_2_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_2_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_3` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_3_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_3_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_3_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_3_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_lens_3_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `condenser_stigmator` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `descan_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `eds_detector_system` | 5 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `image_diffraction_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `mini_condenser` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `mini_condenser_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `mini_condenser_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `mini_condenser_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `mini_condenser_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `mini_condenser_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `objective_aperture` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `objective_lens` | 23 | 8 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `objective_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `objective_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `objective_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `objective_lower_pole` | 15 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `objective_stigmator` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `objective_upper_pole` | 15 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `sample` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3.toml` | `sample_stage` | 7 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `(module)` | 16 | 7 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `ac_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `beam_deflector` | 12 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `c1_c2_pole_piece_cartridge` | 7 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_aperture_2` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_aperture_3` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_deflector` | 12 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_1` | 12 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_1_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_1_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_1_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_1_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_2` | 11 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_2_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_2_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_2_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_2_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_3` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_3_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_3_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_3_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_3_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_lens_3_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `condenser_stigmator` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `descan_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `eds_detector_system` | 5 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_adapter_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_adapter_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_adapter_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_adapter_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_adapter_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_adapter_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_diffraction_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_dp11_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_dp12_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_dp21_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_dp22_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_dph1_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_dph2_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_dsh_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_dstg_quadrupole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_hp1_hexapole` | 10 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_hp2_hexapole` | 10 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_hpol_hexapole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_ish_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_ol_post_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_ol_post_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_ol_post_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_ol_post_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_ol_post_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_ol_post_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_qpol_quadrupole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_sad_plane` | 6 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl11_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl11_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl11_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl11_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl11_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl11_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl12_lens` | 9 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl21_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl21_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl21_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl21_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl21_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl21_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl22_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl22_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl22_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl22_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl22_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `image_tl22_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `mini_condenser` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `mini_condenser_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `mini_condenser_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `mini_condenser_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `mini_condenser_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `mini_condenser_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `objective_aperture` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `objective_lens` | 23 | 8 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `objective_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `objective_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `objective_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `objective_lower_pole` | 15 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `objective_stigmator` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `objective_upper_pole` | 15 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `sample` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ImageCorrector.toml` | `sample_stage` | 7 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `(module)` | 16 | 7 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `ac_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `adapter_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `adapter_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `adapter_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `adapter_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `adapter_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `adapter_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `beam_deflector` | 12 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `c1_c2_pole_piece_cartridge` | 7 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_aperture_2` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_aperture_3` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_deflector` | 12 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_1` | 12 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_1_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_1_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_1_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_1_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_2` | 11 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_2_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_2_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_2_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_2_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_3` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_3_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_3_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_3_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_3_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_lens_3_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `condenser_stigmator` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `descan_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `eds_detector_system` | 5 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `image_diffraction_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `mini_condenser` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `mini_condenser_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `mini_condenser_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `mini_condenser_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `mini_condenser_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `mini_condenser_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `objective_aperture` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `objective_lens` | 23 | 8 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `objective_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `objective_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `objective_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `objective_lower_pole` | 15 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `objective_stigmator` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `objective_upper_pole` | 15 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_dp11_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_dp12_scan_deflector` | 11 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_dp21_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_dp22_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_dph1_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_dph2_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_hp1_hexapole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_hp2_hexapole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_hpc_hexapole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_hpol_hexapole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_qpc_quadrupole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_qph1_quadrupole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_qph2_quadrupole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_qpol_quadrupole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl12_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl12_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl12_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl12_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl12_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl12_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl21_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl21_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl21_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl21_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl21_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl21_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl22_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl22_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl22_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl22_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl22_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `probe_tl22_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `sample` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector.toml` | `sample_stage` | 7 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `(module)` | 16 | 7 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `ac_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `adapter_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `adapter_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `adapter_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `adapter_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `adapter_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `adapter_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `beam_deflector` | 12 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `c1_c2_pole_piece_cartridge` | 7 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_aperture_2` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_aperture_3` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_deflector` | 12 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_1` | 12 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_1_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_1_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_1_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_1_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_2` | 11 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_2_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_2_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_2_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_2_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_3` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_3_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_3_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_3_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_3_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_lens_3_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `condenser_stigmator` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `descan_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `eds_detector_system` | 5 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_adapter_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_adapter_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_adapter_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_adapter_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_adapter_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_adapter_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_diffraction_deflector` | 13 | 3 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_dp11_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_dp12_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_dp21_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_dp22_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_dph1_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_dph2_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_dsh_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_dstg_quadrupole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_hp1_hexapole` | 10 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_hp2_hexapole` | 10 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_hpol_hexapole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_ish_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_ol_post_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_ol_post_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_ol_post_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_ol_post_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_ol_post_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_ol_post_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_qpol_quadrupole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_sad_plane` | 6 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl11_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl11_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl11_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl11_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl11_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl11_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl12_lens` | 9 | 2 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl21_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl21_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl21_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl21_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl21_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl21_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl22_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl22_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl22_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl22_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl22_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `image_tl22_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `mini_condenser` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `mini_condenser_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `mini_condenser_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `mini_condenser_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `mini_condenser_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `mini_condenser_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `objective_aperture` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `objective_lens` | 23 | 8 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `objective_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `objective_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `objective_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `objective_lower_pole` | 15 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `objective_stigmator` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `objective_upper_pole` | 15 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_dp11_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_dp12_scan_deflector` | 11 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_dp21_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_dp22_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_dph1_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_dph2_deflector` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_hp1_hexapole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_hp2_hexapole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_hpc_hexapole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_hpol_hexapole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_qpc_quadrupole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_qph1_quadrupole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_qph2_quadrupole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_qpol_quadrupole` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl12_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl12_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl12_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl12_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl12_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl12_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl21_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl21_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl21_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl21_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl21_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl21_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl22_lens` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl22_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl22_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl22_lens_lower_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl22_lens_upper_pole` | 8 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `probe_tl22_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `sample` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\column\C3_ProbeCorrector_ImageCorrector.toml` | `sample_stage` | 7 | 2 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG.toml` | `(module)` | 7 | 2 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG.toml` | `feg_accelerator` | 17 | 11 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG.toml` | `feg_c1_aperture` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG.toml` | `feg_deflector` | 11 | 2 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG.toml` | `feg_dpa_aperture` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG.toml` | `feg_electrostatic_lens` | 8 | 1 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG.toml` | `feg_extractor` | 8 | 1 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG.toml` | `feg_stigmator` | 9 | 2 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG.toml` | `feg_tip` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml` | `(module)` | 7 | 2 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml` | `feg_accelerator` | 17 | 11 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml` | `feg_c1_aperture` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml` | `feg_deflector` | 11 | 2 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml` | `feg_dpa_aperture` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml` | `feg_electrostatic_lens` | 8 | 1 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml` | `feg_extractor` | 8 | 1 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml` | `feg_monochromator_slit` | 9 | 2 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml` | `feg_monochromator_wien` | 9 | 2 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml` | `feg_stigmator` | 9 | 2 |
| `E:\tem_simulator_v2\configs\instruments\gun\FEG_Mono.toml` | `feg_tip` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\gun\Thermionic.toml` | `(module)` | 7 | 2 |
| `E:\tem_simulator_v2\configs\instruments\gun\Thermionic.toml` | `thermionic_accelerator` | 17 | 11 |
| `E:\tem_simulator_v2\configs\instruments\gun\Thermionic.toml` | `thermionic_anode_aperture` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\gun\Thermionic.toml` | `thermionic_c1_aperture` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\gun\Thermionic.toml` | `thermionic_cathode` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\gun\Thermionic.toml` | `thermionic_deflector` | 11 | 2 |
| `E:\tem_simulator_v2\configs\instruments\gun\Thermionic.toml` | `thermionic_gun_lens` | 8 | 1 |
| `E:\tem_simulator_v2\configs\instruments\gun\Thermionic.toml` | `thermionic_stigmator` | 9 | 2 |
| `E:\tem_simulator_v2\configs\instruments\gun\Thermionic.toml` | `thermionic_wehnelt` | 8 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `(module)` | 12 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `bf` | 11 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `camera` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `df` | 11 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `diffraction_lens` | 2 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `diffraction_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `diffraction_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `diffraction_lens_lower_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `diffraction_lens_upper_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `diffraction_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `diffraction_stigmator` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter` | 5 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_bias_tube` | 2 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_camera_deflector` | 3 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_dynamic_focus_electrostatic_quadrupole` | 3 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_eftem_output_plane` | 3 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_entrance_aperture` | 3 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_multipole_01` | 7 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_multipole_02` | 7 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_multipole_03` | 7 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_multipole_04` | 7 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_multipole_05` | 7 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_multipole_06` | 7 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_multipole_07` | 7 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_multipole_08` | 7 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_multipole_09` | 7 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_multipole_10` | 7 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_shutter` | 3 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_slit` | 5 | 3 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_tapered_prism` | 6 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `energy_filter_zebra` | 12 | 7 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `flu_screen` | 11 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `haadf` | 11 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `intermediate_lens` | 2 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `intermediate_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `intermediate_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `intermediate_lens_lower_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `intermediate_lens_upper_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `intermediate_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `post_projector_detector_chamber` | 2 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projection_chamber_dpa_aperture` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projector_lens_1` | 2 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projector_lens_1_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projector_lens_1_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projector_lens_1_lower_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projector_lens_1_upper_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projector_lens_1_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projector_lens_2` | 2 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projector_lens_2_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projector_lens_2_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projector_lens_2_lower_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projector_lens_2_upper_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `projector_lens_2_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\EnergyFilter.toml` | `selected_area_aperture` | 10 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `(module)` | 12 | 6 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `bf` | 11 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `camera` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `df` | 11 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `diffraction_lens` | 2 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `diffraction_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `diffraction_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `diffraction_lens_lower_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `diffraction_lens_upper_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `diffraction_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `diffraction_stigmator` | 9 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `flu_screen` | 11 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `haadf` | 11 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `intermediate_lens` | 2 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `intermediate_lens_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `intermediate_lens_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `intermediate_lens_lower_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `intermediate_lens_upper_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `intermediate_lens_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `post_projector_detector_chamber` | 2 | 1 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projection_chamber_dpa_aperture` | 9 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projector_lens_1` | 2 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projector_lens_1_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projector_lens_1_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projector_lens_1_lower_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projector_lens_1_upper_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projector_lens_1_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projector_lens_2` | 2 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projector_lens_2_excitation_coil` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projector_lens_2_housing` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projector_lens_2_lower_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projector_lens_2_upper_pole` | 13 | 2 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `projector_lens_2_yoke` | 7 | 0 |
| `E:\tem_simulator_v2\configs\instruments\project_and_recording_system\NoEnergyFilter.toml` | `selected_area_aperture` | 10 | 0 |
