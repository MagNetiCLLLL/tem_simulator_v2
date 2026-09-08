# HAADF DPA clearance — 2026-09-08

The Projection Chamber DPA design opening is now **12 mm diameter (6 mm radius)**
in both recording-system TOMLs. Its 20 mm outer diameter and nominal vacuum
channel remain as configured. The separate 0.2 mm literature-reference diameter
is retained; 12 mm is a user-authorized simulator design, not an OEM dimension.
Differential-pumping performance is not part of this calculation.

For the ordinary `nano_probe` / `diffraction` preset, the first-order angular
transfer at the DPA is about 32.262 mm/rad. The original 0.1 mm radius blocked
the HAADF annulus. A 6 mm radius admits roughly 60.12–185.97 mrad across the
default 32 × 32, 1 nm raster, including a conservative position/azimuth bound.
At 168.8004 mrad the required radius is at most 5.4461 mm, so the new aperture
has approximately 10.17% radial margin over the previous scan's coherent
isotropic support. The remaining high-angle portion is still physically
blocked in this ordinary preset: resizing the DPA does not disable clipping.

The previously acquired Si [110] scan uses a different, saved DPA-crossover
projector setting. It already passes the complete 60–330 mrad HAADF annulus
with the original 0.2 mm diameter aperture. With a 12 mm opening, it continues
to pass that range. These are signed first-order recording maps; they do not
establish clearance including unmodelled high-order aberrations.

Load `si110_haadf_dpa_12mm.toml` to use that calibrated scan configuration with
the enlarged DPA. Its only change from the previous final profile is
`devices.projection_chamber_dpa_aperture.radius_mm`: 0.1 → 6.0. Sample diameter
10 nm, thickness 5 nm, all lens strengths, detector geometry and other controls
are preserved. Loading reports zero skipped fields; see
`profile_verification.json`. Restart the App to load the new instrument defaults;
an older operating profile can still explicitly restore its saved opening.

`clearance_verification.json` records the independent geometry/route checks.
No new 64 × 64 wave acquisition was run for this aperture edit. The previous
images and their archived input geometry remain in their original acquisition
directory and are not relabelled as a new run.
