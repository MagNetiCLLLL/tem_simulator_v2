# STEM detector bank: one propagation per physical dwell

Status: **development integration, not M04 acceptance or M06 product admission**.
The main-window high-accuracy STEM wave product remains unavailable. This
change does not alter source parameters, loosen source-domain checks, fit an
exit source, or remove the production gate.

## Implemented

- `TipWaveRequest.detector_keys` selects several installed detectors in one
  request. Missing, retracted, readout-disabled or duplicate selections fail
  explicitly. The old single-detector argument remains supported.
- The pipeline executes the gun, incident column and specimen once per dwell,
  then propagates in physical Z order through the selected final detector.
  Reading a channel does not cause another specimen interaction calculation.
- Every inserted intermediate detector absorbs its intercepted wave whether
  selected for display or not. Readout precedes that detector's absorption.
  Coincident detector planes are rejected until a physical ordering is defined.
- A multi-detector path cannot skip an energy filter lying before its final
  detector. A detector before the filter still uses the upstream-only path.
- Every channel retains its own incident complex modes, energy, phase carrier,
  axial reference and conditional histories. Incoherent phases and detector
  responses are never added into a fictitious aggregate phase or channel.
- The numerical readout budget is divided between simultaneously retained
  channels. This changes allocation limits, not electron weights. Segmented
  mode continues to keep native wave data in dependency-bound disk stores.
- The scan accumulator returns arrays with axes `(detector, row, column)` and
  explicit channel keys and dwell coverage. Missing intensity and incomplete
  dwell integrals remain NaN. Duplicate samples, changed input identities,
  changed channel sets and invalid weights are rejected before any channel is
  incremented. The accumulator holds only scalar responses, not wave buffers.

Single-detector compatibility: `TipWaveResult.detector` and the two-dimensional
`response_per_tip_electron` array refer to the final detector. They are not a
sum across the bank. New consumers must use `detector_readouts` and
`detector_response_per_tip_electron` for multiple channels.

## Development CLI and archives

```powershell
.venv\Scripts\python.exe scripts\trace_tip_wave.py --profile physical_scan.toml --describe --stop detector --detectors haadf df bf --scan
.venv\Scripts\python.exe scripts\trace_tip_wave.py --profile physical_scan.toml --output outputs\new_scan --stop detector --detectors haadf df bf --scan
```

These are development commands, not a supported default STEM workflow. The
profile must supply a physically admissible explicit tip model and supported
installed operators. The current default has no selected coherence model;
selecting Gaussian-Schell with the unchanged 5 nm / 0.3 eV source is also
outside the current declared domain. Neither command substitutes a source.

`--describe` can inspect an inadmissible configuration and reports
`source_status=UNAVAILABLE`; that is not computation or admission. Actual
execution still rejects it. Source-domain failures now enter the existing
failure-receipt path, including their quantitative report and saved effective
settings, rather than failing before diagnostics are written. The redundant
pre-execution construction of tip modes was removed: execution constructs
them in the physical gun pipeline itself.

Each dwell archive includes a `detectors` list mapping physical detector keys
to their readout and native-wave files. Every intermediate detector exports
its complete complex modes even when phase display is not selected. Final
detector filenames keep the old convention without duplicating that wave
archive. Numeric channel indices are used in filenames, not component keys.

`scan_response.npz` includes:

- `detector_keys`, `physical_rows`, `physical_columns`;
- `detector_response_per_tip_electron`, `detector_dwell_coverage`;
- the old final-detector-only arrays for compatibility.

## Validation and limits

The [focused run](evidence/20260912T133321Z-stem-detector-bank-dd0d724b/report.json)
completed **84 passed, zero failed/errors/skips**, including new detector-bank
cases and adjacent export, segmentation, readout and P0 regressions. Raw
pytest output/XML and input hashes are preserved alongside the report.
The earlier failed test iteration is also retained, not overwritten.

The [additional compatibility run](evidence/20260912T133444Z-stem-bank-compatibility-6a5b44a5/report.json)
completed **28 passed, 3 failed**, with zero errors/skips. Two existing gun
fixtures request the unchanged coherent 5 nm / 0.3 eV monoenergetic source and
are rejected at the declared 1% paraxial-domain budget (bound 6.972%). The
existing material fixture is rejected by combined sampling at 3.63247 radians
per step versus the 0.8*pi limit. These are unresolved failures already present
after P0; they were not deleted, skipped, or made to pass by changing physical
parameters. The broader previous 17-failure suite was not rerun in full here.
All inventoried inputs remained unchanged during both final runs. Changed
Python modules compiled successfully and the working-tree whitespace check
passed. None of those engineering checks qualify a full image.

The bank tests use explicitly labelled upstream stubs to count execution and
exercise the real detector masks, complex readout, absorption and disk store.
The 2-by-2 fixture demonstrates four specimen calls, not twelve, for three
channels. It is **not** a physical-tip-to-Si image or evidence of throughput
on the user's microscope configuration. Phase preservation is tested within
each mode; physical escaping Gaussian tails are included in the probability
balance rather than renormalised away.

The CLI source failure test uses the actual source-domain check and confirms
that input tip parameters are preserved. Separate production-admission tests
confirm that selecting a detector bank does not open the main-window gate.

No full image, visible GUI, GPU, real microscope acquisition, preset fitting,
source retuning, or broad non-paraxial solver qualification was performed.
Unrelated instrument-recorder edits remain unchanged.

## Next physical milestone

Resolve the low-energy emission/propagation applicability problem without
changing saved tip inputs or bypassing extraction and acceleration. Then
execute an actual saved operating point through every installed component,
compare gun/column and specimen numerical refinements independently, and
measure a small real raster at the physical detectors. Only that evidence
can justify opening the corresponding GUI STEM product. M04 and M06 are not
completed by this detector-bank work.
