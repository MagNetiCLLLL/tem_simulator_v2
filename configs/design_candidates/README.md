# Experimental input designs

These are explicitly loaded, input-only Working Point packages. They are not
default operating presets, executed beam caches or qualified imaging results.
Import in **Working Points** and restore only after saving current settings.

`matched_tip_d95_20260915_r500.temwp` contains a 500 nm curvature radius and
0.5911968652 degree emitting cap, matched to the Flat source's projected D95 of
10.0568940617 nm. It retains the Flat Nanoprobe + Diffraction lens values as a
starting point, **not a working curved-source focus**. The paired `_flat.temwp`
keeps the reference inputs. Both contain only a JSON manifest, no numeric arrays.

See `docs/development/tip-curvature-comparison-2026-09-15.md` for measurements,
failed preset acceptance, numerical limitations and reproduction commands.
