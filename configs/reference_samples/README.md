# Reference CIF library

Place standard `.cif` or `.mcif` files in this directory, then use **Sample → Reference CIF → Refresh**. The selected file supplies the atoms, unit cell, elemental composition and occupancy used by the model. **Open CIF** selects a file outside this library. There is no Virtual sample mode; retract the sample for a vacuum calculation.

The default specimen is a **10 nm diameter disk, 5 nm thick**, using `Si.cif` along **[110]**, with [1 -1 0] along X. The Si file is the unmodified file supplied by the user on 2026-09-08 (a = 5.44370237 Å). Existing saved dimensions are preserved on load, and switching references preserves the current specimen dimensions.

An optional sidecar with the same stem, for example `Si.toml`, can specify:

```toml
key = "si_110"
name = "Silicon [110]"
zone_axis = [1, 1, 0]
in_plane_axis = [1, -1, 0]
chemical_symbol = "Si"
template_preset_key = "si_110"
```

Without a sidecar the filename is its key/name, the beam direction is [001], and X is [100]. The numerical template supplies grid defaults only: its old analytic atoms never replace the CIF. A missing reference is reported rather than replaced by another material.

Optional `thermal_sigma_angstrom` requires `thermal_source`; this is a one-axis RMS assumption for frozen phonons, independent of the crystal density. Per-element RMS or a positive manual global RMS can override it. A bare CIF does not automatically supply a thermal displacement model. The supplied Si reference retains the project's sourced room-temperature 0.085 Å assumption; Au uses 0.080 Å. `inelastic_preset_key` explicitly declares a sourced inelastic reference; its elements must agree with the CIF. Bare/external CIFs need explicit inelastic settings.

**Rutherford tail material: From structure** counts symmetry-expanded, occupancy-weighted atoms per unit-cell volume and multiplies by the current interacting thickness. Mixtures retain their individual atomic numbers and scattering contributions. **Screening: Thomas–Fermi/Molière** derives each element's screening angle from Z and beam energy. Manual material/screening choices remain independent overrides and are retained while automatic mode is selected.

The screening angle regularizes forward scattering; it is not the detector's inner angle. Maximum angle is the cutoff of the approximate tail, not a crystal property. The tail begins beyond the isotropic wave-sampling limit and is integrated through the actual detector masks. It is a screened Rutherford approximation, not a full Mott or multiple-scattering solution. Formula reference: [Geant4 electron nuclear scattering, equations 97–98](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/elastic_scattering/elecnuc.html).

Partial and mixed occupancies are supported for bulk composition/Rutherford/EDS. Coherent multislice currently requires a fully occupied, explicitly ordered/disordered supercell and rejects fractional-site CIFs rather than silently selecting only their majority species. MCIF reading supplies atomic positions; magnetic moment scattering is not implemented.

Legacy Si/Au virtual profiles migrate to real CIF references, preserving saved dimensions and explicit manual tail settings. Legacy vacuum profiles retract the holder. Old ideal diffraction spots, rings and virtual density-map controls are retired. An old amorphous-carbon reference needs an actual structure file before it can be simulated.
