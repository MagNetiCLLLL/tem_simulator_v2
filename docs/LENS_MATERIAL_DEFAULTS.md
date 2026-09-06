# Lens material defaults

Research and implementation: 2026-09-06. These are generic design defaults,
not a Titan bill of materials or a measured/heat-treated material certificate.

## Selection and evidence

| Region | Default / retained model | Evidence and boundary |
| --- | --- | --- |
| Magnetic poles | Pure iron, FEMM reference | JEOL explicitly lists pure iron and permendur as polepiece materials. Pure iron is our common starting design, not a claim that all commercial poles use it. |
| Magnetic yokes | Same pure-iron reference | JEOL describes iron yokes. Using the same reference for poles and yokes is an explicit simplification, not a material-identification result. |
| Excitation windings | Existing insulated copper winding | JEOL describes a copper-wire objective coil. Windings remain current sources in nonmagnetic space; no iron curve is assigned to them. No conductivity, coil resistance or temperature model is added. |
| Vacuum / nonmagnetic supports | Existing nonmagnetic model | The default does not turn structural shells, vacuum tubes or an unresolved mixed-material cartridge into magnetic iron. |

Primary sources, accessed 2026-09-06:

- [JEOL: polepiece](https://www.jeol.com/words/emterms/20121023.100857.php)
- [JEOL: yoke](https://www.jeol.com/words/emterms/20121023.104257.php)
- [JEOL: objective lens and copper-wire coil](https://www.jeol.com/words/emterms/20121023.060558.php)

Permendur is a documented alternative for poles, not an automatic substitution
for every objective lens. Different grades and processing conditions require
their own sourced B-H data. No Permendur curve is fabricated or obtained by
rescaling pure iron. User curves and material-class overrides remain supported.

## Numerical data

`configs/materials/lens_defaults.toml` selects `femm_pure_iron` by stable key,
not alphabetic library position. The existing 21-point curve in
`configs/materials/magnetic/femm_pure_iron.toml` is unchanged.

The pinned [FEMM library mirror](https://raw.githubusercontent.com/cenit/FEMM/7d9e8ed90772ddb939a3256f976e231bcb5cca97/bin/matlib.dat),
block `Pure Iron`, supplies both:

- Linear reference: `Mu_x = Mu_y = 14872`, dimensionless relative permeability.
  This constant is used as published in that block, **not** fitted from the
  B-H points. It is a linear design approximation, not a universal measured
  permeability for pure iron and not a saturation model.
- Nonlinear reference: the original B (tesla), H (A/m) pairs. The existing
  piecewise-linear H(B) solver and final out-of-range rejection are unchanged.
  The last B value, 2.56 T, is a data-range endpoint, not an assigned saturation
  induction.

The source file's SHA-256 is
`ba0ca80ffae56440a9ad785c9089b0e7f63b45e1c7a73dd5c0bdc338beeb63b7`.
The hash, all 21 pairs and both constant-permeability entries were checked
against that source during this change. FEMM's
[soft-material notes](https://www.femm.info/doku/doku.php?id=softmagneticmaterials)
are background provenance; they do not identify this reference as an OEM grade.

## UI and persistence

Open **Simulation > Configure lens models...**. Unconfigured lenses now show
the default linear permeability and nonlinear material. **Use default material**
resets only the selected lens's material draft; **Use geometry field** applies
it. Source details and approximation limits are in tooltips.

Existing explicit permeability, B-H snapshots, per-material overrides, coil
inputs, geometry and strengths are preserved. Switching lenses never carries
the previous lens's custom CSV into an unconfigured lens. Linear reference
provenance is saved with its numeric value; editing that value removes the
reference attribution. Changes to the default file do not rewrite saved recipes.

Material defaults do **not** provide ampere-turns at 100%, install missing coils,
generate field recipes on startup, optimise presets, start a solve, clear cached
results or bypass readiness checks. An empty coil input still requires an
explicit value. Global tiers still require all relevant lens recipes, and
geometry/mesh/boundary/nonlinear convergence require separate validation.
