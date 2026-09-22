# User-defined particle cutoff and Ray Diagram extent — 2026-09-19

Custom cutoff calculations, saved full-precision state and dependency-checked
continuation are already connected to the normal particle workflow. This
update adds a narrow spatial coverage strip directly beneath the main Ray
Diagram and checks that its positions follow the displayed calculation.

## Using a cutoff and continuing

1. Open Live tuning and select **Capture current settings**.
2. Enable **Calculation section (optional)**. Select a named plane or use
   **Custom Z** and enter the target coordinate. Participating components are
   selected from the declared ranges when live adjustment is needed; a
   one-shot section calculation does not require any ranges.
3. Run the particle calculation at the desired quality. High accuracy also
   honours this section request. An accepted completed particle calculation
   is saved automatically; **Saved particle state** shows the completed
   endpoint, available restart endpoint, quality, population, save time and
   actual file path. A failed write is not reported as a successful save.
4. Select a later cutoff and calculate again. A matching executed upstream
   state is reused. **Load saved section...** supplies that state in a later
   session without overwriting current instrument settings. Changed upstream
   fields, source, model or numerical inputs invalidate the affected reuse.

Supported user-defined coordinates are on the straight column from the gun
exit to the active axial endpoint. With an enabled energy filter, that range
ends at the filter entrance. Arbitrary stops inside the gun or along the bent
filter path are not implemented. A cutoff before a finite specimen/support's
actual transport exit is rejected rather than discarding part of that
interaction. Vacuum participation can conservatively require recomputation.

## Display behaviour

- The 8-pixel track shares the ray plot's physical Z mapping and follows
  panning, zooming, axis reversal and resizing. It is not a time percentage.
- Green marks the completed axial interval of the displayed result; grey is
  the remainder of the visible axis. This describes executed transport, not a
  claim that every electron survived to that plane.
- A purple marker marks the current requested cutoff. Moving the request or
  the observation cursor never advances the completed interval.
- A blue marker distinguishes the available restart plane when it differs
  from the completed endpoint. This is an executed checkpoint, not proof that
  a disk save succeeded or that the current upstream settings still match.
- Amber identifies an earlier result after physical inputs change, including
  an intermediate live-tuning frame with a newer request pending.
- Off-view markers become directional arrows while their actual coordinates
  remain in the text/tooltip. The summary is selectable and copyable.

The completed position is read only from accepted result metadata accompanied
by a section checkpoint. Display tails, current controls and filter-local
coordinates cannot invent completed coverage. Material exit checkpoints can
reside separately from incident checkpoints, so the interval is not incorrectly
clipped back to the specimen. Historical results without endpoint metadata
remain readable with their calculated range explicitly unavailable.

## Verification

Final verification passed **139 related cases in two runs**: 133 range-display,
section UI/controller/archive, incremental drawing, display-cache, axis,
hidden-panel and live-dock cases in 187.467 seconds, plus all 6 parameter-source
and pending-main-window-frame cases in 22.83 seconds. There were no failures,
errors or skips in those final runs. The 133-case receipt is
`tmp/ray-cutoff-20260919/cleaned-regression.xml`. Existing Pydantic deprecation
warnings and a pyqtgraph shutdown disconnect warning remain. This is focused
UI/controller verification, not a new full-repository or physical qualification.
`compileall` completed successfully for `src` and `tests` after testing.

The new coverage reader/widget/workspace tests distinguish requests, completed
coverage, retained restart planes and stale frames. They include zoom, pan,
inverted axes, resizing, coincident endpoint markers, off-view arrows,
missing historical metadata and a changed request after recapturing settings.

An extended GUI run initially stalled when an unrelated hidden panel was shown.
Instrumentation showed no increasing bar refresh, paint or plot-update counts;
instead, thousands of earlier tests' Qt objects were being destroyed together.
The test teardown now explicitly flushes Qt's deferred deletion after pytest-qt
closes the test widgets. Production scheduling was not changed to mask this
test-lifecycle problem. The parameter-source fixtures were also updated to copy
the current shared catalog dependencies and select the default recording module;
the pending live-frame test now checks the range strip's freshness and endpoints.

Screenshots under `tmp/ray-cutoff-20260919/` use explicitly artificial UI
fixtures to check placement and text, not to qualify physical propagation.
Five states were inspected at 1600 by 950 pixels: request only, completed versus
requested, stale, zoomed with off-view markers, and coincident completed/cutoff.
The track is 8 pixels high; the 46-pixel widget includes the copyable summary.
This change adds no numerical execution or array hashing to GUI updates.
No coherent development, application restart, commit or push is included.
