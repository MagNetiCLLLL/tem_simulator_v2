# Physical tip editing and model switching

## Editing workflow

Select **Cold Field Emission Tip** and open **Physical Layout → 3D Parts**.
The component panel and Model Inspector now navigate to this same workspace.

- **Dimensions** edits the installed geometry: apex curvature radius, cone
  half-angle, shank length and saved emission-cap half-angle. Save applies the
  shared `configs/sources/FEG_tip.toml` definition through the existing atomic
  assembly validation/reload path.
- **Tip model / emission…** edits active tip emission and selects the physical
  curved-tip or Flat tip (planar) model. Geometry is read-only in this dialog;
  its dimension button returns to the physical editor. These emission changes
  are operating overrides, not automatic writes to shared TOML defaults.
- **Fit emitting surface** fits the actual apex cap at equal spatial scale.
  Gold faces are the emitting surface; the rest of the metal does not emit.
  Face/edge selection highlights curvature radius and cap-angle parameters.
  A differing operating override is shown beside its saved default, read-only.
- An unsaved/invalid geometry draft must be saved or reverted before a different
  source edit can replace active state. Navigation preserves that draft.

Saving dimensions retains active emission and numerical values that differ from
the previously loaded defaults. Other values follow the new saved defaults.
Prescribed total current stays fixed; prescribed flux density stays fixed while
total current changes with emitting area. An incompatible active cap and new
cone geometry rejects the save and rolls back the file and live state.
Geometry reloads also retain the C3 aperture opening and Camera/Flu screen
insertion choices. Explicit assembly installation still applies its defaults.

The emission colour is face metadata on one closed solid. It does not split
the tip into open material bodies, create a new material assignment, or change
the geometry used by copying and solid operations.

For radius R and emitting-cap half-angle theta, the patch has projected diameter
`2 R sin(theta)`, depth `R (1 - cos(theta))`, and curved area
`2 pi R^2 (1 - cos(theta))`. The default R = 100 nm and theta = 10 degrees give
diameter 34.73 nm, depth 1.5192 nm and area 954.56 nm². The cap angle defines
*where* electrons originate; the maximum angle from each local normal defines
*their directions*. The cone angle is a third, independent geometric quantity.

## Model switching and Ray Diagram

Flat tip is the default for new desktop sessions, headless default states,
standalone FEG construction, and explicit FEG / FEG + Mono assembly installation.
The shared `configs/sources/FEG_tip.toml` declares
`default_tip_emission = "flat_tip"`; both linked gun manifests carry the same
default. This is applied by the source/assembly loader, not a GUI-only override.
Thermionic assemblies retain their thermionic source.

The curved cap/cone recipe remains available for explicit selection. Its saved
geometry is not the active Flat tip emission law. To select it, open
**Tip model / emission…** and enable **Curved tip (off: Flat tip)**.
Geometry-only reloads and saved profiles/checkpoints retain an explicit source
choice; they do not reset to Flat tip. Existing profiles and calculated results
are not rewritten. Switching models changes the calculation/cache identity.

No scalar source distribution, current, electrode voltage/reference, lens
strength, extraction/acceleration stage, aperture or vacuum participation is
retuned by this default-selection change. Coherent development remains paused.

Turning curved-tip mode off selects the historical planar emission / analytic
gun field / legacy integrator combination. It is not an otherwise identical
calculation with curvature removed. Prescribed current and launch distributions
can differ, and existing lens settings are retained without automatic matching.

Apply validates a detached candidate for the entire gun. For example, a
tip-referenced gun-lens voltage cannot be passed into the historical analytic
model, which requires the extractor reference. Invalid edits leave the source,
optics and displayed result unchanged; no voltage reference is silently changed.

Accepted edits invalidate pending jobs and request a fresh particle Preview.
The previous Ray Diagram is marked as awaiting recalculation. Completed diagrams
identify the source from their captured input state. A failed recalculation does
not turn the previous result into a result for the newly selected source.

Gun trace cache identity now includes the full executed component geometry,
aperture/deflector/stigmator positions and exit plane, separately from profile
serialization (which deliberately defers geometry to TOML). Source/model,
electrostatic field and vacuum dependencies remain included.

## Bounded particle comparison

The normal Preview request reconstruction and full gun/column particle path
were exercised with 49 displayed rays, 1 mm column integration step, default
assembly/presets, and unchanged lenses. No specimen wave calculation was run.

| Quantity | Curved tip | Historical planar |
| --- | ---: | ---: |
| Prescribed tip current | 100 nA | 10,000 nA |
| Gun survivors (including diagnostic support probes) | 38 / 49 | 49 / 49 |
| Weighted gun transmission | 0.770833 | 1.0 |
| Weighted sample-plane transmission | 0 | 0.510204 |
| Principal column stop | Column wall | C2 aperture |

The curved Preview's one surviving central support ray has zero statistical
weight. It reaches the downstream BF absorbing detector in the regression test,
but is not evidence of nonzero transmitted sample current. Coarse Preview
sampling is not a converged current estimate. This comparison verifies model
selection, cache separation and real interception; it does not qualify matching
or imaging performance for either source.

## Input designs and historical results

The supplied `particle_tip_30mrad_20260915.temwp` is a historical curved-tip
input-only design with no computed arrays. It remains readable, but its older
gun manifests and cell schema do not match the current defaults, so direct
application is rejected. No package rewrite or implicit migration is performed.
Input-only packages with compatible schemas and matching dependencies can load
under the current solver and must be recalculated. Bundled input paths can
relocate only when their archived content matches (LF/CRLF differences in
TOML/CIF are allowed). Actual input changes are rejected; archived identities
are untouched.

Calculated checkpoints retain the strict solver/content compatibility gate;
tagging a package with retained arrays as input-only does not bypass it.
No downstream source, artificial transmission, new wave calculation or automatic
voltage-reference conversion is introduced by this update.

## Flat-default validation

Focused checks cover both FEG variants, standalone/headless/GUI startup,
thermionic preservation, explicit curved selection, source-editor controls,
profile and snapshot round trips, cache separation, geometry reloads, grouped
sampling, and physical gun aperture interception. Curved-model tests select
their curved fixtures explicitly instead of depending on application defaults.
The source controls remain in the single emission editor. No long coherent or
high-accuracy image calculation was run for this default-selection change.
