# Current validation snapshot

Recorded 28 September 2026. This summarizes the completed classical diagnostic
development; it is not full microscope qualification or a new validation of
historical coherent calculations. Original detailed reports, failed attempts,
receipts and arrays remain local; published history retains older documents.

## Scope and evidence

The required Phase00–08 work completed its declared scope: field/execution
identities, isolated diagnostic workers, measured display optimization,
read-only Hardware tuning feedback and independent Virtual electrons sessions.
The primary tip-origin source, physical forces, particle integration and main
calculation archive format were preserved. Optional perturbation execution and
sampling an electron from a main result were not implemented.

Seven accepted local scopes contain **1,593 distinct passing tests**, with no
skipped mandatory cases. Repeated runs, early failures and intentional failing
child processes are excluded from this total.

| Scope | Passed | Accepted run ID |
| --- | ---: | --- |
| Classical contracts | 267 | `f5c71c09ed7146168932223ff6667e43` |
| Gun fields | 121 | `1dbb444945c8417084c1c26ed2dd7bf8` |
| Electron execution | 477 | `4f3506bc1a0f4d90b0fbf9ea21f9bb13` |
| Field UI | 428 | `803e00844bff4dac82548ae22b6d1f83` |
| Particle continuation | 124 | `62f6f1395fbc4ff69e839745574e71b2` |
| Performance observation | 78 | `6ae13330f9934ce884c2ae6e18ebdb5d` |
| Acceptance policy | 98 | `44b6f7200a9340f5a29eb529da720d2e` |

Evidence index: `outputs/agent-validation/20260928-phase08/summary.json` (local,
excluded from Git). Each receipt records its source/input hashes, command,
environment, JUnit/log paths and unchanged source during execution. The work
started from `04e87584ecb88a802813e2f109d67a5719615399` with local changes. Windows
Python 3.12.3 and Qt offscreen were used; numerical jobs ran serially with one
thread. Classical, gun-fields and particle-continuation passed before final
archive-reader/provenance hardening; the four other scopes were rerun afterward.
Those earlier receipts are not relabelled as tests of a later source snapshot.

An isolated installed wheel passed startup outside the checkout with `-I`.
Its nine-particle optical calculation included tip emission, extraction,
acceleration, gun focusing, apertures and transport to 3026.4 mm. A separate,
labelled uniform-magnetic-field diagnostic completed 100 reference steps and
round-tripped all eight float64 trajectory arrays exactly and immutably. Both
new panels opened offscreen. This did not validate specimen/detector signals.
Wheel SHA256: `055314c3e92ff282823f02d60f73ab30314aae87a6246cbcf1c77f45702abe50`.

## Numerical and performance limits

The bounded field comparison completed 18 reference paths across flat, curved
and near-aperture cases. Maximum energy-invariant drift was 3.64e-8 eV against
the declared 1e-6 eV fixture bound. Linear-residual and curved launch-potential
checks passed. One particle per case and two refinement levels do not establish
full-population convergence or interchangeable short/extended electrostatic
domains. Its status is `COMPARISON_COMPLETE_REVIEW_REQUIRED`; domain equivalence
remains `NOT_ESTABLISHED`.

The controlled display-only comparison used identical inputs and a 984×940,
DPR1 offscreen canvas. Median input-to-paint improved from 438.95 ms to
343.91/333.52 ms in two candidate runs (21.7–24.0%); median painting fell from
37.02 ms to 6.31/6.36 ms. All 35 completed results retained exactly the same
settings, termination metadata and eight physical arrays. Viewport simplification
is bounded to 0.1 device pixel and never modifies transport/checkpoint arrays.
This is a rendering improvement, not faster numerical integration.

Fresh final measurements were separate observations, not another matched
speedup comparison:

| Case | Measured result |
| --- | --- |
| Flat axial diagnostic | 105.90 ms warm request median |
| Curved-tip diagnostic | 2.817 s warm request median, reference solver |
| 49-particle gun | 3.178 s warm; 23.19 ms exact reuse; no repeated transport on hits |
| Continuous electron edits | 363.67 ms median; 381.61 ms p95; 19.89 ms exact reuse |
| 49-particle continuation, 460–470 mm | 1.068 s; terminal arrays exactly match direct reference |

The continuation test was before the specimen and reused the executed gun
prefix. Saving/loading its 1,512,652-byte archive took 699.59/962.30 ms. During
80 rapid slider changes the final interaction run had zero blank frames, but
no geometry matched the latest input before the next edit; the final exact
settings appeared 256.17 ms after release. This is not fully real-time tracking.
The 35 final diagnostic results still matched the earlier candidate arrays
exactly. Generated evidence remains under the local Phase05/Phase08 directories.

## Current boundaries

- Hardware feedback observes retained calculations and explicitly marks stale
  or unavailable data; selecting an observation does not run upstream transport.
- `.temdiag` loads diagnostic histories, never a configurable production source
  or continuation checkpoint. Recalculation in current fields is explicit.
- Coherent tip-to-column development remains paused. Full microscope
  qualification remains `UNQUALIFIED`.
- Native desktop/OpenGL, GPU, 5000-particle performance, remote CI and the full
  project suite were not established by these CPU/offscreen checks.
- Scientific mechanisms and qualitative trends are the target. Experimental
  agreement and instrument-time savings have not been measured by these tests.

## Repository cleanup checks — 28 September 2026

The subsequent cleanup changes documentation, development utilities and tracked
artifacts; it does not change runtime numerical models. It removes 1,330 old or
generated files from the current Git tree, including 25 obsolete standalone
scripts. Tracked Markdown decreases from 184 to 38 files. Tracked file content
decreases from approximately 244 MiB to 15.3 MiB; published Git history is not
rewritten and its storage is not reduced by this change.

Obsolete documents/scripts/reports were moved to a local ignored archive with
SHA256 verification. Generated outputs, temporary files, reference downloads
and UI screenshots remain locally and are no longer tracked. Current inputs,
profiles, reference provenance, source, tests, CI and installation assets remain
available. Eight retained historical links point to their exact pre-cleanup
Git version; the remaining relative Markdown links resolve within the tracked
tree. Preserved reference URLs are indexed in `references/SOURCES.md`.

Actual cleanup validation:

- **119 tests passed** for environment setup, acceptance contracts, assembly
  identities, packaged reference inputs and calculation-artifact fallback.
- **98 acceptance-policy tests passed**, receipt
  `b13917f9157e4f1c9ac0bd622921ec52`; no skipped mandatory cases and no source
  changes during execution. These are 217 distinct cleanup checks, separate
  from the earlier 1593-case development snapshot.
- A clean export of staged files, with no old `outputs/`, `tmp/` or screenshot
  directories, built a wheel successfully. That wheel was installed into an
  isolated prefix and passed `installation_diagnostic_smoke.py` with `-I`
  outside the checkout, including the bounded classical column, diagnostic
  persistence and offscreen panels.
- `git diff --cached --check` passed. No generated arrays, caches or diagnostic
  sessions are included in the new commits.

Local evidence: `outputs/agent-validation/repo-cleanup/`; archive inventory and
checksums: `tmp/repository-cleanup-20260928/plan.json`. Cleanup wheel SHA256:
`10729f8eb4d92ee2608221c20832879d86864c8173c245179efb155c342c259e`.
These checks do not rerun paused coherent work or establish native desktop/GPU
or experimental acceptance.
