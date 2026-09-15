# Continuous classical tip curvature

## Scope

One classical source spans flat and curved emitting surfaces. The operating
parameter `curvature_nm_inv` is κ in nm⁻¹; κ = 0 is the existing flat source and
R = 1/κ is derived for κ > 0. Fresh installations still start at zero. No lens
presets, gun voltages or assembly positions are automatically changed.

The same projected truncated Gaussian, local angular law, kinetic-energy law,
ray identities, current and weights apply at every κ. With the current 5 nm
FWHM, projected D95 remains 10.056894 nm. The tip centre/apex stays at
**(0, 0, 0)**. Every point keeps its original x/y, while off-axis points bend
upstream to **negative Z**. Local emission directions rotate with the surface
normal. The centroid of the curved emission patch is not re-centred to Z = 0.
No electrode field, launch energy, current or projected source size changes
with κ. Subsequent paths and interception can change with launch position and
direction.

This is `analytic-tip-centred-curvature-v3`: the existing analytic gun field and Boris
transport are used throughout. Extraction, acceleration, focusing, bores and
apertures remain active. It is **not** a self-consistent electrode-field solve
for the deformed metal surface, a tunnelling-current prediction or a newly
qualified coherent TEM/STEM source. Historical electrode-field and coherent
models remain separate advanced options and are never silently converted.

## Use

1. Select **Cold Field Emission Tip** in the component list. Its quick controls
   include **Tip curvature (0 = flat)**, initially 0, with a 1e-8 nm⁻¹ increment.
2. For a slider, capture current settings in **Live tuning**, add **Cold Field
   Emission Tip / Curvature**, and start live tuning. The visible initial range
   is 0 to 1e-5 nm⁻¹; both endpoints are editable. Slider tracking submits changes
   while held down, using the existing completed-frame/latest-value scheduler.
3. Use Preview or Medium while adjusting. Perform high accuracy separately
   when desired. Tuning does not overwrite completed specimen signals.
4. Save an operating profile or input working point to retain the chosen κ.
   Ordinary geometry refresh preserves it; explicit assembly reset restores 0.

The tip emission editor accepts κ and shows the emitting surface, centre Z = 0,
edge Z, local-normal arrows, derived radius and support diameter. The
historic metal CAD remains an assembly reference,
not a claim that its electrode field was re-solved for the operating patch.

## Geometry and validity

In consistent length units, let r² = x² + y² and c = sqrt(1 − κ²r²).
The surface is z = −κr² / (1 + c), with normal (κx, κy, c). The rationalized
expression stays stable near zero curvature. An orthonormal rotation from +Z
to this normal maps local outgoing directions. Both sides have negative Z,
not opposite signs of Z; their radial normal components have opposite signs.
The sphere centre is (0, 0, −R), distinct from the tip apex at the origin.
There is no surface-area correction or angle-dependent energy rescaling.

The full projected support radius is a = 3 FWHM / 2.354820045. Inputs require
κa ≤ 0.95 and a downstream local angular support. Invalid values are rejected,
not silently clipped. K − eφ is conserved relative to each launch point in
analytic-field tracing (the current flat default has launch potential zero).

## Persistence and validation

Nonzero κ and its geometry version enter the serialized gun and cache identity.
Changing it invalidates dependent ray and signal products; restoring zero can
reuse the matching flat cache. Launch normals and directions are retained for
source-angle colouring. Operating-profile restoration validates coupled source
fields together; an absent historical curvature means zero.
Profile format 11 explicitly saves the curvature model. Historical v1 gun
records, nonzero-curvature profiles (version 9 or earlier), and old snapshots
keep the earlier sag-plus-direction model. Angle-only v2 records, including
format-10 profiles without a discriminator, retain their planar launch positions.
They do not silently acquire the new geometry. To explicitly switch an old
record, use **Use continuous tip · start flat** and then enter the desired
curvature. The tip editor labels historical models; all three model/cache
identities remain distinct.
The launch-potential energy reference has its own solver identity in both gun
and staged caches; older results remain readable but are not reused as newly
executed propagation under that revision.

Focused tests cover geometry, local-angle/current/energy invariance, small-κ
stability, support probes, persistence, cache dependencies, actual gun transport,
full Preview ray calculation, source editing and slider updates before release.
No long coherent-wave calculation is started.
The current centre-anchored v3 affected regression passed 87 tests in 44.70
seconds. This includes symmetric negative sag, an exactly fixed centre,
preserved projected coordinates/current/local angles/launch energies, actual
gun history starting at the curved surface, Preview execution and v1/v2/v3
profile/snapshot/cache compatibility. UI checks use offscreen Qt. This does
not qualify a self-consistent electrode field or coherent imaging.
The angle-only v2 affected regression passed 83 tests in 49.53 seconds,
including exact fixed launch positions, unchanged energy/current, actual gun
propagation, slider updates, and explicit v1/v2 profile/snapshot compatibility.
Affected modules compiled successfully; UI validation was offscreen.
The preceding v1 regression run passed 92 tests (offscreen Qt, 56.34 seconds);
affected Python modules also compiled successfully. This is automated/offscreen
validation, not a live user-session or hardware acquisition check.

A preceding v1 49-ray comparison with the pre-change tracer reproduced the default flat
exit x/y, slopes, energies, weights, IDs, survival flags and full x/y histories
exactly. At κ = 1e-8 and 2e-8 nm⁻¹, maximum exit-x changes were 0.093514 and
0.187027 nm; at 1e-5 nm⁻¹ the change was 93.510859 nm. All 49 rays reached the
gun exit in each case. This checks local continuity, not full preset focus or
integration-step convergence.

The broader interactive-bank regression has four existing failures, reproduced
using the pre-curvature interactive-calculation module: a legacy wave fixture
missing `beam_voltage_kv`, an unregistered `EnergyFilterResult` in bank snapshot
replay, and two tests attempting wave imaging behind the paused coherent-source
admission gate. Those restrictions are not bypassed by this feature.
