# Match transport result publication repair

Clicking **Match transport** could finish its candidate checks and then fail
with `CalculationResult.signatures must be a dictionary`, before the candidate
reached the GUI commit gate. The user's lens controls therefore remained at
the previous working point for this failure.

`CalculationManifest` freezes its signature mapping to preserve provenance.
The transport result producer passed that read-only mapping directly to
`CalculationResult`, whose current contract requires a dictionary. The producer
now copies the signature entries into a dictionary when building the display
result. Manifest immutability, strict result validation, accepted lens controls,
and reuse of the refined forward pass remain unchanged. No compatibility
fallback or solver modification was added. A scan of all seven production
result constructors found no other instance of this mismatch.

## Verification

- Both new regression cases reproduced the exact error before the repair:
  `tmp/match-transport-before-20260925.xml` (2 failures).
- After repair, 33 tests passed with no failures, errors or skips in 46.030 s:
  `tmp/match-transport-after-20260925.xml`.
- The new tests use explicitly synthetic particle paths and candidate proposals,
  with real snapshots, transport measurements, working-point checkpoints, result
  construction, commit/undo, worker signals and Qt result readouts. They verify
  reuse of the final forward result and independent display/manifest signatures.
- Existing tests cover the Match transport button route, cancelled and zero-current
  requests, clipping, result contracts, transaction state and controller ownership.
- Affected files compile and `git diff --check` passes. The project interpreter
  imports the repaired source directly, including without a `PYTHONPATH` override.

This is a data-interface regression repair, not a new physical qualification.
No full physical matching calculation or complete repository suite was run.
The running user application was not restarted. No commit or push was made.
