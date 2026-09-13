# FEG source editor and calculation admission

## Reproduction

With the installed default assembly, apply the grounded surface model in
Model Inspector and request High accuracy with STEM wave imaging enabled.
Both classical-surface and coherent-surface selections survive rejection and
remain selected when the editor is reopened. The calculation is rejected by
the existing wave-source admission boundary; no automatic source reset was
reproduced. Closing the editor without Apply still discards its draft.

The classical-only message previously described coherent surface transport as
unimplemented even though a separate development route exists. It now explains
that the selected classical source supplies no coherent phase. Rejection also
explicitly states that source selection and previous results are unchanged.

## Changes

- Move the compact `FEG tip...` action to the existing top control row.
- Show concise, distinct availability messages for classical and coherent
  surface selections. Extended context remains in a tooltip.
- Fix the near-field launch from an installed assembly: its immutable mapping
  proxies cannot be copied with generic `deepcopy`. Use the complete instrument
  graph codec, preserving runtime optics without reloading the manifest.
- Keep preview edits detached from the active source and completed images.

No source parameters, physical transport, admission thresholds, production
cache policy or historical result data were changed. Restart the application
to load these GUI changes; the existing user application was not terminated.

## Verification

Offline, input-bound receipts (91 tests across three completed groups):

- [65 compatibility checks](evidence/20260913T111800Z-tip-source-compatibility-eeae1da6/report.json).
- [11 editor and rejection checks](evidence/20260913T112020Z-tip-source-editor-scoped-9a89d937/report.json).
- [15 near-field integration checks](evidence/20260913T112030Z-tip-source-near-field-scoped-0c1fcecd/report.json).

The viewer handoff recorder in the new editor test is an offline UI fixture,
not a wave calculation. Existing near-field integration tests execute their
small numerical cases; they do not validate a full source-to-image chain.

[Before-change failures](evidence/20260913T111049Z-tip-source-ui-before-9544cf70/report.json)
are retained. The initial full-graph preview assertions also detected the
existing one-ULP aperture-anchor normalization, not a source reset. Source
ownership checks now start from that canonical serialized layout without
relaxing numerical tolerances or changing production normalization.

Two combined GUI runs stalled and were stopped, not counted as passes:
[first combined run](evidence/20260913T111303Z-tip-source-ui-after-3172c3ec/report.json)
and [diagnostic run with stack dump](evidence/20260913T111740Z-tip-source-ui-diagnostic-80e195b1/report.json).
The diagnostic sampled background signature preparation and a Qt event wait;
the cross-test waiting issue is not resolved by this patch. This is not a
full-suite pass. The completed groups above were run independently.

[Offscreen layout rendering with Windows fonts](evidence/20260913T111303Z-tip-source-ui-after-3172c3ec/model-inspector-fonts.png)
confirms a content-width button in the existing row, not a standalone full-width
row. This does not establish validation of the running desktop session.

## Remaining physical boundary

The classical curved source has particle emission and gun transport. Selecting
the coherent reservoir instead produces a wave boundary, which the current
classical Ray Diagram pipeline cannot sample as if it were the old particle
distribution. Its executed complex fields still need a dedicated display route.

Full coherent TEM/STEM admission remains closed. The existing executed gun
comparison gives 29.2522 versus 58.1572 nA at identical physical inputs when
only a numerical coordinate width changes; relative complex L2 difference is
1.239. This is failed convergence, not a UI selection failure. See
[the numerical report](JOINT_SURFACE_GUN_2026-09-13.md) for the unresolved basis,
boundary, aperture and column issues and remaining full-chain integration.
