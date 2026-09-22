# Scientific scope and functional equipment names

The simulator represents its own declared mechanical and optical structure.
Its current goal is physically correct interactions and qualitative parameter
responses. Matching a commercial instrument's current settings, magnification
table or absolute performance is not an acceptance requirement.

Instrument records can suggest topology and coupled interactions. A response
sign from another instrument is not transferable until axes, winding polarity,
mode and held controls are defined. Imported numbers are reference observations,
not authoritative simulator calibration for a different structure.

## What a qualitative check must specify

Declare the parameter varied, controls held fixed, observation plane, coordinate
frame, operating range and predicted relationship before running a sweep.
Distinguish a held optical system from one that automatically refits a working
point after each edit. Changing detector readout selection must not change
physical interception, absorption or propagation.

| Subsystem | Bounded physical expectation | Important conditions |
| --- | --- | --- |
| Axisymmetric magnetic lens | Reversing field polarity reverses Larmor rotation while retaining radial focusing. Increasing field magnitude increases the isolated paraxial focusing power. | Compare the same field shape, energy and axes. This is not a claim that spot size or full-column magnification is globally monotonic in lens current. |
| Independent stigmator channels | Normal and skew quadrupole components permit continuous principal-axis rotation; reversal of one pure component exchanges its focusing axes. | Define the two field bases and compare at the same plane; upstream rotation changes detector-axis appearance. |
| Dipole / scan deflection | In the linear range, reversing a kick reverses its incremental position/angle response; scaling a held drive scales that incremental response. | Fixed optics and held coupling, no clipping or saturation; do not silently recalibrate. |
| Aperture | Enlarging the same centered opening cannot remove additional trajectories from a fixed incident population. | Fixed source, upstream optics and sampling; total transmitted current is the observable, not normalized image brightness or RMS spot size. |
| Hexapole corrector | Nonlinear kicks and induced aberrations follow the multipole symmetry and vary continuously near a fixed working point. | Reference aberration conventions must match. There is no universal claim that increasing every corrector control improves the probe. |
| Pixelated detector | With a fixed detector model, counts scale with incident dose in the linear range; spreading redistributes signal, read noise and saturation affect measured counts. | Do not confuse optical blur with detector blur, correction maps with forward gain, or DQE with electron absorption probability. |
| Energy filter | Energy dispersion, energy-dependent interception and slit selection follow the executed filter trajectory. Narrowing nested slit openings cannot increase transmitted current for the same incident population. | A detector before the entrance does not traverse the filter. A requested downstream path must traverse it. Slit width is not alone a certificate of energy resolution. |

These are validation requirements, not new results claiming that all subsystems
have passed them. Unit consistency, conservation, numerical convergence and
independent limiting cases remain required. Coherent tip-to-column work remains
paused; these scope and naming changes do not start wave calculations.

## Naming and evidence

Use functional names such as Pixelated camera, Screen camera, Hexapole probe
corrector, Symmetric objective, Electrostatic beam blanker, EDS detector array
and Post-column energy filter. CCD or CMOS should be specified only when that
sensor distinction is actually part of the model.

Current configuration explanations and validation messages use scientific
terminology. Bibliographic URLs retain exact source addresses. Internal legacy
keys remain stable so historical profiles and dependencies stay readable.

The instrument recorder presents scientific labels, but routes requests using
the original hardware identifiers and saves the original metadata. Its displayed
record details are a labelled presentation copy, not rewritten acquisition data.

This change does not import instrument calibration values or alter geometry,
fields, currents, detector sampling, apertures or source parameters.

## Validation of this change

- Compared all 32 configuration TOMLs before/after: 56 name or descriptive-source
  string changes; all other values, identity keys and reference URLs match.
- Compared the changed core modules after removing string literals from their
  syntax trees: no physical algorithm or non-text value changed.
- Selected assembly, persistence, recorder, blanker and orientation checks:
  133 passed. Another 24 component-operation cases failed during fixture setup
  because the temporary recording document omitted its subassembly files.
  The same setup error was reproduced in an isolated copy of the preceding
  committed code, before these naming edits. The full suite was not run.
- Offscreen main-window and recorder presentation checks passed, including
  distinct camera labels, unchanged original identity data and error/progress
  labels. This is an offline GUI check, not a live microscope acquisition.
- Python compilation and Git whitespace checks passed.

No complete physical trend qualification is claimed by these naming checks.

The subsequent [first qualitative-science batch](qualitative-first-batch-2026-09-18.md)
repairs the component-test fixture errors above, adds primary parameter meanings,
and records bounded classical trend results and an energy-handoff correction.
Its [parameter inventory](qualitative-parameter-inventory-2026-09-18.md) separates
active inputs, numerical controls, historical metadata and model-specific limits.
