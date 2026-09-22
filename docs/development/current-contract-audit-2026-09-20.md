# Current interface and persistence audit - 2026-09-20

Status: completed. Final current-test coverage is **5,858 passed, one skipped,
zero unexecuted and zero remaining failures**, from the complete suite plus
affected-module retests against unchanged production code.

## Scope

This audit follows the request to simplify the current program instead of
maintaining old-version execution adapters. It covers source, script callers,
GUI dispatch, operating profiles, complete state, calculation results and local
particle archives. Original instrument records and historical files are not
rewritten. No Git commit or push is included.

The scientific target remains qualitative responses for the simulator geometry.
Classical emission still traverses extraction, acceleration, focusing and
apertures. Coherent tip-to-column development remains paused. Local wave operator
tests do not qualify production coherent illumination or the whole microscope.

## Current contracts

- State schema 78, operating profile 12 and particle archive 2 accept their
  declared current fields. Unsupported schemas and removed fields fail explicitly
  before activating them. Readers do not silently rename devices, manufacture
  missing ray identity or reinterpret downstream-source parameters.
- Component keys are validated against the current topology. Duplicate or
  incomplete detector and corrector records are rejected. Detector insertion
  and electronic readout remain independent.
- Sample orientation has one stored quaternion. Euler angle controls derive
  from and update that quaternion; they are not a second orientation state.
  Five retired specimen controls were removed from active state and callers.
- Scan calibration uses named current reference planes. Derived coil coupling
  is not a second independently editable or persisted input. Virtual layout
  markers do not expose nonphysical kick controls.
- Operating-profile application validates a candidate before committing changes.
  Saving refuses a mixture of one assembly selection and another installed
  instrument. Saving is atomic and does not overwrite an existing file on a
  validation failure.
- The energy filter has ten actual multipole carriers, one energy-window owner
  on its slit, and explicit child component records. Loading or reading an
  existing state does not reapply acquisition-mode defaults. Installing the
  filter makes its participation follow the requested physical path; there is
  no independent hidden switch that bypasses installed optics.
- Particle archives contain the actual executed state and dependency identities,
  with bounded array loading. They do not create a configurable source or
  recalculate missing upstream transport on load. Numerical products remain
  local and excluded from Git.

## Behavioural defects repaired

- Fixed unreachable cache clearing, including completion of an in-flight build.
- Fixed stale/reentrant GUI result publication and current callback signatures.
- Removed a second, hidden sample-page EDS interface and its unused dispatch.
  The dedicated EDS page remains the active entry.
- Source colour rotation now uses emission records joined by actual particle
  identifiers. Missing source information stays unavailable rather than being
  inferred from downstream positions or array order.
- General alignment evaluates a transmitted calibration beam on detached copies.
  The live instrument's blanking state is preserved on success and failure.
  Assembly-triggered condenser recalibration uses the registered alignment
  definition and publishes only its two condenser settings; replacing the
  solver's candidate tree cannot publish its temporary emission budget.
- Structured profiles retain the complete ten multipole field/calibration
  records, tip emission quadrature and every accelerator stage's operating
  controls. Mechanical positions remain owned by the installed geometry.
- Energy-filter control edits no longer repeatedly reset mode defaults.
  Explicit software energy-window requests move the physical slit; complete
  profile records retain the actual blade positions and calibration.
- Fixed quaternion-to-Euler conversion near the +/-90 degree chart poles.
  Boundary and 1,000 random quaternion/negated-quaternion round trips retain
  the rotation matrix within 2e-15 absolute error.
- Corrected the slit centre-loss readback to convert the complete displacement
  difference to energy, including a nonzero zero-loss offset and signed
  dispersion.
- Current profile script callers no longer expect a returned list of silently
  skipped fields. Old field assignments that had no active consumer were
  replaced with current calls.
- Removed the unused effective-axis implementation and generic detector upgrade
  path. Finite material propagation requires actual material-path metadata.

## Validation method

The initial whole-suite run collected 5,671 cases: 5,451 passed, 209 failed,
10 errored and one skipped. Failures were separated into production defects,
obsolete test calls/fixtures, explicitly unsupported coherent source requests,
and boundary expectations no longer reachable in the current geometry.

Changes to tests preserve what they can establish. Bounded local wave tests use
an explicitly supplied local probe. Production coherent requests test rejection
before execution, rather than presenting a fabricated gun-exit source as a
qualified tip-origin chain. Current condenser and projector tests distinguish
reachable settings from explicit bounded failure with rollback.

The CUDA mask discrepancy was traced to the CPU test reference normalising each
surviving diffraction pattern to unit mass. The reference now uses the same
Parseval normalization and verifies transmitted mass; the original numerical
tolerances were not relaxed. Five focused CPU/CUDA checks passed.

A separate actual CPU/CUDA comparison exposed cumulative complex64 FFT norm
roundoff. Each GPU Fresnel propagation now controls its norm against the
incoming norm times the retained band fraction. It does not restore physical
bandpass, material or aperture losses. The original detector-loss comparison
keeps rtol=2e-4 and atol=2e-7. Thirty final focused checks passed. In the new
32-step test the maximum relative norm errors were 4.9738e-7 without a band
cutoff and 3.2225e-7 with real band loss, below the declared accumulated
float32 roundoff bound 32*epsilon = 3.8147e-6. Complex fields remain complex64.

Full regression runs use the project virtual environment, isolated offscreen
Qt settings and independent local caches. Four shards use two CPU threads each,
with nested numerical libraries limited to one thread. No running user GUI is
restarted or terminated. Static call checking resolves only provable Python
signatures; it does not replace runtime validation of dynamic calls.

## Final evidence

The final complete suite collected 5,859 cases across 458 test files. The first
full execution finished with 5,842 passed, 16 failures and one skip. Those 16
failures were obsolete test setup or expectations. Only the affected tests were
then updated; the production code remained unchanged. Complete affected-module
retests passed 25, 94 and 98 cases respectively.

A fresh collection and per-test evidence ledger reconcile the current 5,859
cases to **5,858 passed, one skipped, zero unexecuted and zero remaining
failures**. This is complete regression plus complete affected-module retests,
not a claim that one invocation ended with every test green.

| Execution | Cases | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: |
| Final shard 0 | 1,742 | 1,740 | 2 | 0 |
| Final shard 1 | 540 | 539 | 0 | 1 |
| Final shard 2 | 1,794 | 1,786 | 8 | 0 |
| Final shard 3 | 1,783 | 1,777 | 6 | 0 |
| Affected-module retest 0 | 25 | 25 | 0 | 0 |
| Affected-module retest 2 | 94 | 94 | 0 | 0 |
| Affected-module retest 3 | 98 | 98 | 0 | 0 |

The one skip is
`test_direct_alignment.py::test_fixed_lens_aperture_scaling_matches_recalculated_toml_metrics`:
its stored absolute scaling reference predates movement of the C2 aperture.
Current trend and constraint checks remain active; that historical numerical
reference is not presented as validated.

The common 581 production source, configuration, script and entrypoint files
remained byte-for-byte unchanged during this final full execution and retesting.
The SHA256 of their sorted path/hash mapping is
`8196eaa7e4fb9ae46c088ef5edb69c952fdae13bb2611ed82d17c9912fcf4ab6`.
The original input inventory is `tmp/project-contract-final-source.json`.
An initial shard-0 collection-only error executed zero cases; its receipt remains
in `tmp/contract-final-shard-0-collection-error/`. The corrected shard 0 completed
all 1,742 cases recorded above.

The final static audit parsed 1,011 Python files and resolved 18,778 explicit
calls, with no syntax or definite argument/keyword binding errors. This checks
provable imported functions and constructors; dynamic dispatch, generated
arguments and runtime overrides require the runtime coverage above.
`compileall` and `git diff --check` passed after the final tests completed.

The [machine-readable validation record](current-contract-audit-2026-09-20.json)
retains the exact counts and production identity. Local execution receipts are
in `tmp/contract-final-shard-{0,1,2,3}/` and
`tmp/contract-final-retest-{0,2,3}/`; each contains its collection, per-test
results, XML and final exit record. The final current collection, accepted
per-test ledger and summary are in
`tmp/current-contract-accepted-{collection,ledger,summary}.json`.
Generated numerical caches and array outputs remain local and excluded from Git.

## Using the updated application

Restart the application to load the updated code. The existing user GUI was
left running. Active persistence accepts the current schemas only; unsupported
old fields and schemas are explicitly rejected without automatic conversion.
Original historical files and acquisition metadata were not rewritten.

Coherent development remains paused. This audit establishes the tested current
software contracts and bounded numerical behaviour; it does not certify the
entire physical microscope chain or commercial-instrument performance. No new
large full-instrument performance benchmark or visible desktop frame-rate
measurement is claimed in this round.
