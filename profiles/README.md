# Detached particle gun design

`particle_tip_30mrad_20260915.temwp` is the complete **input-only** design that
passed the declared classical particle checks. It has no cached rays or waves.
Import it through **Working Points > Import... > Restore working point**.
The ordinary default assembly and extractor-relative gun voltage remain unchanged.

Use Preview to inspect the layout/rays. To reproduce the nominal numerical
budget, set **High-accuracy rays = 10369** and **Step (mm) = 0.025** in the toolbar.
Do not apply a lens preset or reload the default assembly after restoration.
This point is implementation-pinned and cannot be blindly restored after solver
changes. Full conditions, limits and scalar evidence are in
[the matching report](../docs/development/GUN_MATCHING_CONTINUATION_2026-09-15.md).

The two `particle_tip_surface_30mrad_769_*_draft.toml` files are older exploratory
drafts. They do not carry the final gun geometry or convergence qualification.
