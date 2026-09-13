# Classical particle tip geometry

Coherent tip-to-column propagation is paused. This change exposes the existing
curved-surface particle model; it does not qualify a new coherent image source.

## Use

1. Open **Model Inspector > FEG tip…**.
2. Select **Use curved-tip particles**. This changes the editor draft only.
3. Edit the apex curvature radius, cone half-angle, shank length, emitting-cap
   half-angle, prescribed current and normal/tangential kinetic energies.
4. Select **Apply**, then **Update rays** for particle tracing. Save an operating
   profile to retain edited values; opening the editor does not rewrite TOML.

The reference remains `configs/sources/cold_feg_tip.toml`: 100 nm apex radius,
5 degree cone half-angle, 1000 micrometre shank, 10 degree emitting-cap half-angle,
100 nA current, and 0.2/0.1 eV mean normal/tangential energies. These are reference
model inputs, not a newly calibrated microscope or tunnelling-current prediction.
Old planar particle profiles and coherent profiles are not silently converted.
The wave preview button is disabled while this work is paused.

## Parameter meanings

For spherical radius R and emitting-cap half-angle theta:

| Quantity | Definition |
| --- | --- |
| Apex curvature | 1/R; controlled by the physical radius |
| Cone half-angle | Angle of the metal shank relative to the axis |
| Emitting-cap half-angle | Polar angle at the sphere centre; geometric, not electron divergence |
| Projected patch diameter | 2R sin(theta) |
| Patch depth | 2R sin(theta/2)^2 |
| Apex-to-edge arc length | R theta, with theta in radians |
| Emitting surface area | 4 pi R^2 sin(theta/2)^2 |

The interface shows both degrees and the derived radian value. The highlighted
cross-section uses equal length scales; arrows indicate local surface normals,
not computed trajectories. Only the apex detail is shown, with the shank cropped.

Particles are sampled uniformly in surface area. Their velocity is outgoing
relative to each local normal, with normal and tangential energy components
sampled from the existing positive exponential distributions. Total energy,
energy width and local angular distribution are derived from those components.
Small-cap sampling now uses half-angle identities to avoid cancellation to a
zero-size planar source. No phase, virtual gun-exit source or extra broadening
is added. Curved launch positions and full direction vectors reach the existing
relativistic particle transport and grounded extraction/accelerator field.

Geometry is included in the field and ray-cache identities. Changing the
emitting area alone invalidates emission/rays but does not unnecessarily solve
an unchanged metal/electrode boundary. Changing physical tip geometry changes
the field request. Inactive historical source values are not added again.

## Local-only calculation data

Generated arrays/checkpoints under `docs/development/evidence`, `outputs`,
`tmp` and `.temsim-wave-cache` are local-only. 114 previously tracked numerical
array files (645,815,099 bytes) were removed from the Git index; all local files
were retained. Existing published history was not rewritten. Reports and input
settings remain versioned. Acquisition records under `instrument_records`,
reference data under `data`, and test input fixtures are not globally ignored.

## Validation

46 focused checks passed in 51.32 seconds, including an actual nine-particle
surface-to-gun-exit calculation, local normals/energy/current, narrow-cap limits,
edited profile/snapshot/cache identity, invalid input handling and offscreen GUI
checks. The editor was also rendered and visually inspected. Changed Python
files compiled. No long coherent wave calculation or new TEM/STEM image
qualification was run. Raw engineering receipt:
`evidence/20260913T225207Z-curved-tip-particle-geometry-b4bf2e04/report.json`.
