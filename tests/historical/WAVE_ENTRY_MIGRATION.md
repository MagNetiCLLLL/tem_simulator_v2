# Historical wave-entry tests and HANDOFF v2

Three baseline tests used synthetic incident ray bundles/statistics to call
the formerly available specimen-plane pupil implementation:

- `test_a5_state_profile_cache_and_production_stem_phase`
- `test_tem_reference_cif_frozen_phonons_average_intensities_on_finite_roi_grid`
- `test_stem_atomistic_frozen_phonons_average_detector_intensities`

Their original versions are retained in Git baseline
`5eac9855ff8eefa2a3d68ddba3029f7c23e03b0c`. The initial broad-audit source-gate
failures remain in the HANDOFF ledger and are not rewritten as historical
passes.

The current tests first assert that production rejects the synthetic source.
They then override admission **only inside their individual pytest fixtures**
to retain the existing algorithm checks: A5 changes phase rather than amplitude,
profiles/cache identities preserve the coefficients, and frozen-phonon outputs
average intensities rather than coherent amplitudes. Names explicitly say
`isolated` or `historical`; physical parameters, expected values and tolerances
remain unchanged. No production gate, GUI, CLI or profile admits this override.

These tests do not demonstrate the new gun-to-specimen, coil-driven STEM or
post-specimen detector adapters, which remain separate unfinished requirements.
