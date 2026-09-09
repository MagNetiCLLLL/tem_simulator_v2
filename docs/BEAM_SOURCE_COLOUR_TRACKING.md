# Beam source colour tracking

## Meaning

**Source position** colours label the azimuth of each emitted position about
the original emitted-bundle centroid. They do not encode velocity direction,
energy, intensity, or scattering type. The reference is captured once, before
aperture losses; changing Z never recentres or recolours the survivors.

Each trajectory carries an immutable `source_ray_id` and
`source_azimuth_rad`. Ordinary propagation preserves both. Detailed specimen
transport selects the parent metadata using the terminal electron's original
incident-column index, including compact, reordered and repeated descendants.
Absorbed or backscattered electrons do not acquire forward detector paths.

Weighted loss-channel representatives can share an ancestor. This is lineage
of a numerical source sample, not a claim that one physical electron splits
into several electrons. Preview, Medium and High accuracy use different source
samplings; colours are not unique global identifiers across separate runs.

**Interaction type** remains a separate display mode. X-ray and event colours
retain their category meaning; auxiliary quadrature photons are not assigned
invented electron identities. Grey electron paths indicate undefined source
azimuth (for example the central ray) or unavailable legacy lineage.

## Views and caches

- Ray Diagram, Sample Interactions 3D and the right-hand beam panel have
  independent **Source position** (default) / **Interaction type** selectors.
  In source mode the same
  known ancestor has the same colour in all three views.
- Ray Diagram and Transverse X-Y prefer the matching, completed detailed
  specimen-exit checkpoint. A valid empty checkpoint stays empty.
- Without a valid detailed exit, the ordinary downstream branches are labelled
  as an optical reference, not as the detailed specimen result.
- Transverse X-Y includes all selected exit groups. Its bounded representative
  subset is selected before clipping, so moving Z does not replace blocked rays
  with different rays just to fill the display budget.
- Sample Interactions 3D preserves source metadata on incident, material and
  exit paths. Switching colour mode only redraws the cached scene.
- Publishing a detailed result refreshes Transverse X-Y at the same Z and
  scale, or queues that refresh until a hidden panel is shown.
- Detailed rays show the captured beam position. Optical-reference raster
  offsets are not applied to them; independent STEM image playback is unchanged.
  The downstream reference-only interaction budget is also withheld rather
  than reported for a different electron population.
- Legacy incident caches recover source identity from their retained gun
  history. Compact legacy scattered branches without ancestry remain grey;
  they are never matched to source electrons by their compact row number.

The transverse pattern-rotation diagnostic is a spatial correlation relative
to the source. It includes image inversion and deformation and is suppressed
when correlation is weak. It is not the magnetic-field Larmor integral.

Colour metadata does not participate in propagation, multislice, scattering
probabilities, ray weights, detector integration or TEM/STEM image formation.
No expensive specimen calculation is needed merely to change a display mode.

## Selected-plane analysis

Use **View** in the right-hand beam panel to choose a function. Double-click
an axial plot or use the existing Z selector to inspect another cached plane.

| View | What it measures |
| --- | --- |
| Position X-Y | A bounded sample of reaching trajectory positions in µm. |
| Angular X-Y | Projected ray directions θU and θV in mrad. |
| Angle distribution | Weighted polar-angle histogram relative to +Z. |
| Beam intensity | A 64 × 64 map of weighted flux crossing the plane. |
| Interaction breakdown | Source fraction and current per recorded channel. |
| Phase space U / θU, V / θV | Position versus direction; useful for convergence and focus. |

Spatial and angular coordinates follow the Ray Diagram projection. Source
colours do not rotate independently of the labelled reference wheel. Each
view retains its own range while Z changes; **Fit beam** explicitly fits the
active view. Trajectories may extend outside the visible range.

**Interaction type** uses circles for primary/zero-loss populations, triangles
for elastic channels, diamonds for inelastic channels and stars for the
recorded elastic + inelastic combinations. These categories survive subsequent
projector focusing. Unknown channels remain unknown. Combined finite-exit
categories describe the existing independent elastic × inelastic approximation;
they are not new correlated collision histories. Channeling is not inferred
from a focused pattern, crystallographic orientation or a final ray angle.

Intensity maps, histograms and interaction tables use **all reaching weighted
rays**, not the bounded scatter display sample. At an exact interception plane
the incident flux is included; it is excluded at later planes. Source and
branch weights are applied once, and neither clipping nor viewport selection
renormalises the survivors. Hover over a map or histogram for bin values.

Calibrated results report pA per bin; otherwise the unit is percent of source
per bin. The summary always distinguishes total plane flux from visible flux.
Ambiguous or invalid probability metadata disables quantitative readout while
retaining valid geometry. The intensity colour maximum updates per plane;
compare numeric values, not colours, across planes. Changing bin area by
zooming also changes flux per bin; this is not current density per unit area.

These are cached **geometric electron flux** diagnostics, before detector
acceptance and response. They are not detector counts, coherent wave
intensities, TEM images, STEM images or X-ray spectra. No field propagation,
specimen interaction or imaging calculation is rerun when switching views.
One full selected-plane dataset is shared across the analysis functions and
invalidated when a new result is published, including an updated detailed exit.

## Focused validation

- `tests/test_beam_plane_data.py`: weighted interpolation, exact clipping,
  compact detailed branches, empty/stale results, unit validation and histogram
  conservation without viewport normalisation.
- `tests/test_beam_analysis_modes.py`: all seven functions, projected angles,
  source/interaction styles after focusing, per-view fixed ranges, all-ray
  intensity, hover values, shared plane caching and publication invalidation.
- `tests/test_beam_tracking_modes.py`: recorded elastic/inelastic categories
  and conservative handling of unknown metadata.

- `tests/test_ray_identity_display.py`: boundary continuity, fixed clipping
  reference, projection rotation, compact/repeated descendants, empty/stale
  checkpoints and stable display subsampling.
- `tests/test_ray_source_colour_mode.py`: bounded ray colour groups, validated
  downstream routing, display-only mode switching and result publication.
- `tests/test_detailed_ray_presentation_guards.py`: no reference-only budget or
  mismatched raster animation on detailed specimen exits.
- `tests/test_sample_interaction_source_colours.py`: source lineage throughout
  the local scene and display-only mode switching.
- `tests/test_downstream_transport.py`: terminal ancestry selection and
  unchanged physical coordinates, weights and metrics.
- `tests/test_segmented_column_cache.py` and
  `tests/test_calculation_manifest_artifacts.py`: cached source identity,
  immutable metadata and legacy incident restoration.

A default-assembly 49-ray vacuum preview also verifies zero colour changes
from source to specimen and across the specimen boundary. This is a display
identity check, not a full high-accuracy specimen or image validation.
