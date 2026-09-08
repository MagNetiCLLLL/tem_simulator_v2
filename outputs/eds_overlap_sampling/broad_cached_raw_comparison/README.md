# Broad-beam EDS-only overlap comparison

The disabled estimator returns zero because none of the original 7,635 arriving rays intersects the 10 nm specimen. EDS-only weighted overlap integration returns a small positive expected spectrum while preserving every original electron trajectory.

| EDS overlap setting | Material candidate rays retained | Conditional material mass | Expected detected counts | Additional EDS time |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 0 | 0 | 0 | 0.007 s |
| 256 requested points | 194 | 4.1804090e-6 | 0.0226388471 | 4.13 s |
| 1,024 requested points | 787 | 4.2396633e-6 | 0.0229196817 | 15.33 s |

The 256-point total is 1.2253% below the 1,024-point result in this case. This checks numerical integration sensitivity for the selected approximation; it does not establish that the inferred KDE beam density or photon angular quadrature is converged. Scott KDE introduces Gaussian tails and smooths source structure that the original ray sample does not resolve.

The assumed baseline exposure contains 15,238,059,263.82 emitted electrons and 7,756,172,165.28 reaching the sample plane. The predicted mean remains only about 0.023 detected photons: a Poisson acquisition would still have about **97.7% probability of zero total counts**. These are expected-count results with Poisson sampling disabled; a nonzero estimate does not guarantee an observed Si photon.

This is a disclosed **hybrid diagnostic**, not the user's recovered complete state. It combines actual App cache `f4c443…` final incident phase space with `outputs/haadf_dpa_clearance/si110_haadf_dpa_12mm.toml`, a Si disk of 10 nm diameter and 5 nm thickness, vacuum support, 300 kV, and a 68.0% objective setting in the baseline assembly. The actual cached objective setting and EDS dose are unknown. The cached reference Z of 1679.2 mm is reanchored to the intact baseline sample/assembly Z of 1599.2 mm; the sample is never displaced relative to its pole pieces. Raw X/Y coordinates are retained, including their centroid offset.

All three EDS evaluations reuse **one original elastic transport**, computed in 21.29 s. Its zero material tracks and terminal-array SHA256 `455e57f260a52a9034374005782cfb5372f409d979e734f8917fa5a4d9859f80` remain unchanged. EDS auxiliary paths use distinct quadrature identities and full incident-current weights; they do not become downstream/STEM electrons. Total numerical elapsed time was 40.80 s.

`diagnostic.json` includes dose, field diagnostics, guards, KDE parameters, weights, photon transport, timings and source hashes. `*_spectrum.npz` contain unmodified expected-count arrays. `expected_spectrum_comparison.png` is an unsmoothed step plot of those arrays. `original_cached_sample_plane.npz` preserves the original sample-plane cache before Z reanchoring; `incident_footprint.npz` contains the actual arriving raw positions and conditional weights.

Reproduce from the project root, choosing a fresh output directory:

```powershell
.venv\Scripts\python.exe scripts/benchmark_eds_overlap_sampling.py --output outputs/eds_overlap_sampling/a_new_run
```

The archived script and baseline profile are included. This original command also requires the referenced local application-cache manifest/arrays and project configuration; it does not run unchanged in a fresh checkout without that cache. The required original sample-plane data are preserved in `original_cached_sample_plane.npz`, but the script does not yet expose that file as an alternative input. Default installed-detector photon quadrature order is 1; no claim of photon-angle convergence is made. The broad-ray cache alone cannot reconstruct the original application state.

The three `*_equivalent_profile.toml` and `*_equivalent_state.json` exports reconstruct the recorded baseline plus actual field/sample/sampling overrides. All three profiles reload with **zero skipped values**, correct 68.0% objective, 1599.2 mm sample Z and 10 × 5 nm Si dimensions. Loading them in the App will calculate new optical rays; reproducing these exact spectra requires the separately saved cached phase space. `input_Si.cif` is byte-identical to the source (SHA256 `944c5c81df5d813d051f96102eb9be37fac4b7647c2a9d1492d6333122be42c7`).

`artifact_verification.json` records all spectrum/terminal/profile/CIF checks. The recorded acquisition hash for `eds_signal.py` intentionally differs from the later current file: quadrature identity provenance was added after the run without changing the estimator mathematics. The original hash has not been rewritten. Recreate the verification and equivalent profile exports with `.venv\Scripts\python.exe scripts/verify_eds_overlap_benchmark.py`.
