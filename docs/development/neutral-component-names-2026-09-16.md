# Functional component names

Bundled assembly names and current UI text use functional terminology:

| Previous label | Current label |
| --- | --- |
| NanoPulser | Electrostatic beam blanker |
| Iliad / Iliad Ultra | Energy filter / Post-column energy filter |
| Zebra camera / detector | EELS camera |
| Zebra camera deflector | EELS camera deflector |
| MultiEELS | Multi-window EELS |
| ULTRA-X EDS | EDS system |

The storage files are now
`configs/instruments/beam_blanker/ElectrostaticBeamBlanker.toml` and
`configs/subassemblies/energy_filter.toml`. Catalog and composition references
point to these files. Component trees, Physical Layout, energy-filter controls,
runtime names and diagnostic messages use the same terminology.

Old `NanoPulser` assembly selections are accepted by the current catalog and
resolve to the functional name. Reading the selection of a captured assembly
also recognises its old file path without rewriting that captured assembly.
Historical snapshots and archived baseline reports remain unchanged.

Stable module/component keys, serialized Python class identities and internal
field names are retained for compatibility with working points, physical
parameter references and vacuum anchors. They are identifiers, not equipment
display names. Original scientific citations and source URLs also remain in
the configuration metadata so the geometric assumptions remain traceable.

Comparison against the archived stage-3 inputs checked all 30 configuration
TOMLs: only 28 display-name/description or file-reference values changed.
Every other value, including geometry, material, voltages, emission, detector
sampling, placement constraints and source citations, is identical. The
comparison receipt is `evidence/neutral-component-names-20260916.json`.

Validation: 105 targeted tests passed (80 assembly/navigation/blanker tests and
25 energy-filter/EELS/aperture/GUI tests). These cover all 30 assembly
combinations, legacy blanker selections and captured paths, stable component
identity and placement, and the renamed component controls.
