# Simulation model levels

The **Simulation** menu sits beside File and View. These choices select column
lens physics, not the number of rays, integration step, CPU/GPU backend or the
High accuracy product request. New desktop windows start in Ideal Optics;
legacy states and operating profiles without this setting use Custom so their
previous mixed models are not silently replaced.

| Menu choice | Implemented behaviour |
| --- | --- |
| Ideal Optics | Existing finite-field paraxial column transport at reference energy. No empirical spherical kicks, hexapole terms, additional Objective Cc kick or higher-order wave-lens coefficients. Configured defocus remains. |
| Analytical Field | Existing parameterised fields and configured aberration model. Registered spatial maps and geometry recipes are retained but inactive. |
| Linear Geometry Field | All enabled column round lenses require explicit linear FEM recipes. Missing recipes or unsupported geometry fail; no automatic analytic fallback. |
| Nonlinear Material Field | Explicit sourced B-H recipes, jointly solved static axisymmetric fields. No hysteresis or thermal feedback; unsupported/missing inputs fail. |
| Coupled Multiphysics | Disabled. Circuit/thermal/structural feedback is not implemented. |
| Custom / Per-lens Models | Existing per-lens analytic, geometry and imported-map configuration, including legacy behaviour and its explicit approximation warnings. |

## Teaching boundaries

Ideal Optics is a finite-length first-order lens model, not a newly calibrated
thin-lens model or an assertion of zero beam width. It retains focusing, magnetic
rotation, first-order quadrupole/stigmator controls, physical apertures and
detector interception. Pole/material geometry is not used to solve a magnetic
boundary-value problem in either Ideal or Analytical mode.

Only column optical propagation uses reference-energy focusing in Ideal mode.
Ray energy values are not overwritten: source, finite-specimen scattering, EDS,
EELS and local specimen magnetic flights retain their own physical models.
The electron-gun calculation and Energy Filter's dedicated dispersive transport
are not replaced by this column-lens selector. It does not switch TEM/STEM modes,
enable wave imaging, change sample materials or erase scattering signals.

The Ideal wave-lens coefficients are explicitly zero except configured defocus.
They are labelled disabled by model, not fitted or measured. A field-derived
aberration request saved in another model is not executed while Ideal is active.
Linear Geometry selects the column field source; the existing Manual versus
Field-derived wave-aberration choice remains explicit. A geometry field alone
does not certify the TEM/STEM image's wave-aberration model.

## Controls and state

- The menu is exclusive. Unsupported tiers are visibly disabled. Linear Geometry
  becomes selectable when every enabled round lens has a recipe and no declared
  saturation insert requires an unsupported solver. This is a readiness check,
  not a mesh-convergence or geometry-overlap certification.
- **Configure lens models...** opens Model Inspector. Explicit field/model edits
  there, or importing/clearing a map, select Custom and retain current operating
  values. This avoids editing a recipe that the active global tier would ignore.
- Unconfigured lens drafts use the sourced pure-iron [material defaults](LENS_MATERIAL_DEFAULTS.md):
  constant reference permeability for Linear and the full B-H table for Nonlinear.
  Coil ampere-turns remain explicit; defaults do not bypass setup requirements.
- Nonlinear Material Field similarly requires B-H recipes for all enabled round
  lenses. See [static nonlinear fields](STATIC_NONLINEAR_MAGNETICS.md) for reference
  materials, joint-current routing and the distinction from multiphysics.
- Each mode retains round-lens excitation percentages/polarities, field recipes,
  probe/image coefficient options, the additional chromatic-kick setting and the
  existing equivalent-image-lens option. Its first visit inherits current values;
  subsequent visits restore its retained values. No cross-model strength matching
  or lens-preset optimisation is performed.
- Geometry, installations, enabled hardware, source, apertures, specimen,
  detector controls and numerical resolution are shared, not copied into model
  shelves. A shelf cannot restore an old mechanical layout.
- The current model is shown in the status bar and completed-calculation status.
  Model changes invalidate pending workers immediately; previous plots remain
  explicitly stale until a new result is accepted. Other complete results stay
  in the existing bounded result cache.
- State schema 76 and operating-profile format 3 persist the selection and
  shelves. Profile formats 1 and 2 remain readable. Imported arrays are not
  embedded in TOML; their original file descriptors retain the normal provenance
  and geometry checks.

## Cache and numerical verification

The active model participates in stage/result identities and the propagation
plan signature. Inactive shelves are excluded from calculation identities, so
returning to unchanged settings can reuse their old result. Shared physical
geometry changes still invalidate affected products; retaining results never
means accepting an incompatible field map. The persisted solver identity is
`temsim-solver-2026-09-static-bh-v1`.

`tests/test_simulation_modes.py` checks selection, invalid-mode rejection, profile
and State round trips, inactive-shelf identities, map bypass/restoration, actual
linear FEM routing, coefficient suppression, first-order superposition,
reference-energy achromatic column propagation, and menu/cache behaviour.
Synthetic test inputs are not OEM dimensions, coil ratings or material data.
These tests do not establish real-microscope accuracy. Nonlinear numerical tests
and their boundaries are documented separately; multiphysics is not implemented.
