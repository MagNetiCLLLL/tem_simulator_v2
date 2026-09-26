# Virtual electron dock, responsiveness and field display — 2026-09-26

## User-facing changes

**Virtual electrons** is now a native main-window dock, available from
**Ray Diagram → Magnetic field → Electron trajectories → Electrons…** or
**View → Virtual electrons**. It can tabify with Instrument setup or Live tuning,
float, and participate in named layouts and restart restoration. Closing it
retains parameters, cached paths and the overlay. A narrow dock stacks the list
above the scrollable editor; wide panels use two columns. Reopening does not
force a larger dock width or change instrument inputs.

Electron curves use 1.0 logical pixel strokes, or 1.4 for the selected electron,
without the former dark halo. Magnetic lines use 75% opacity in electron mode.
Magnetic arrowheads use fixed screen dimensions, avoiding long spikes caused
by transverse display enlargement. Line colours/density and physical field
values are unchanged. Fit uses the combined visible magnetic/electron extent
with one physical scale. A manually fixed narrow viewport can still clip a
field line; new results do not silently reset that viewport.

Field plots obtain their horizontal bounds from the Ray Diagram ViewBox's
actual global screen positions. Both the spatial view and 2D tesla profile
track resize, layout, dock, sidebar and scroll changes. Spatial Z/U linking
remains optional; the 2D profile still has a field-strength Y axis. Projection
and navigation reuse projected paths and do not integrate particles again.

## Execution and cache ownership

Captured electromagnetic-field preparation and single-electron integration
run in one persistent hidden Python process per controller. The unchanged
full E+B solver remains responsible for extraction, gun focusing, acceleration,
magnetic components, apertures and other supported physical interception.
This change does not simplify the fields or relax numerical accuracy.

The parent retains shared numerical-job admission while the child executes.
The child is limited to one numerical CPU, including its native-library pools.
The prepared scene stays in that process; subsequent parameters send only
electron settings. Existing exact-settings/result caches remain active, so
selection, renaming, display choices and identical duplication do not retrace.
Opaque scene identities prevent reuse after scene replacement or process loss.
Cancellation/stale results are not accepted. Closing the application first
cancels field/particle work, then terminates the diagnostic process once the
other close guards accept shutdown.
On Windows the child uses the base interpreter with the virtual-environment
launcher binding, preserving the project interpreter prefix and dependencies
while avoiding an intermediate redirector process. Tests compare the PID
reported from inside the child with the owned process PID and verify actual
child exit, including cancellation during integration. Numerical imports are
initialized before the Windows pipe-reader thread starts.

## Measured responsiveness and numerical equivalence

The reproducible benchmark is `scripts/benchmark_test_electron_responsiveness.py`,
run with the project Python interpreter and one numerical CPU on the 32-logical-
CPU host. It measures a 10 ms Qt timer during real default E+B transport without
painting, rather than claiming a displayed frame rate. Two fixed cases retain
the same 1 mm maximum step, 20,000-step budget, 1e-4 relative tolerance and 1 pm
position tolerance. Initial energy is 0.3 eV at the physical tip.

| Case | Thread p95 / maximum timer gap | Process p95 / maximum timer gap | Thread / process solve time |
|---|---|---|---|
| 3.0264 m, on-axis | 21.63 / 39.75 ms | 16.02 / 16.26 ms | 7.71 / 6.85 s |
| 550 mm, 5° polar / 45° azimuth | 24.35 / 38.92 ms | 15.98 / 17.84 ms | 4.19 / 3.75 s |

Idle timer gaps were approximately 16 ms on this Windows test setup. There
were no gaps above 50 ms in either bounded benchmark. The process isolates
GUI interpreter contention. These individual runtimes vary with host load and
do not establish an integrator speedup. Cold process startup took 0.349 s,
followed by 1.873 s for first captured-input imports and field preparation,
versus 0.126 s for warm in-process preparation. Later edits reuse the process
and scene; their combined first-use cost was approximately 2.22 s.

All six stored arrays in both cases are bitwise identical: XYZ, momentum,
time, path length, kinetic energy and electric potential. Termination reasons,
step counts and static-energy invariant errors also agree. Local reports and
arrays are under `tmp/test-electron-responsiveness-thread-20260926/` and
`tmp/test-electron-responsiveness-process-final-20260926/`, including
`array-equivalence.json`. These are generated data and remain excluded from Git.

## Native-window verification

An offscreen MainWindow integration executed two actual diagnostic E+B paths:
the full 3.0264 m on-axis path and 550 mm at 5°/45°. One field preparation and
two integrations used the same child process and scene. Duplicate/overlay,
docking, floating, redocking and resizing added no execution requests. At seven
axial positions, the maximum horizontal discrepancy was 2.28e-13 logical
pixels. An additional readiness query confirmed that the child-reported PID
matched the owned process PID and retained the project virtual environment.
That actual child exited on accepted window close. This fixture disabled the
constructor's operating-preset solve and main instrument-population preview;
it is real diagnostic transport, not full-instrument validation.

The report and inspected screenshots are in
`tmp/virtual-electron-dock-live-20260926/`. A separate stored-trajectory replay
checked display-only changes in `tmp/virtual-electron-dock-display-20260926/`.
Scientific scope and unsupported-field stops remain those documented in the
[virtual electron guide](magnetic-test-electron-2026-09-26.md). Coherent work
remains paused, and specimen/detector interactions remain excluded only from
this detached diagnostic.

## Final regression

The final affected-feature run completed with **332 passed**, zero failures,
errors or skips, in **178.143 s**. It covers the particle solver, E+B scene,
field display/geometry, navigation, process isolation/lifecycle, controller,
workspace layouts, both electron/live-tuning docks and CPU resource admission.
The new process suite contributes 13 checks, including cold startup and the
actual Windows child PID. The final command used the project interpreter,
`QT_QPA_PLATFORM=offscreen`, explicit pytest-qt loading and a two-CPU test-process
cap; diagnostic execution retained its one-CPU limit. The original one-CPU
test environment exposed a CPU-fixture assumption that a child could receive
two CPUs despite inheriting an OpenMP limit of one. The clean two-CPU rerun
passed that fixture without changing production CPU-budget code.

Evidence: `tmp/virtual-electron-dock-response-final.xml` and
`tmp/virtual-electron-dock-response-tests.log`. Changed Python modules compile,
and `git diff --check` passes. This is an affected-feature suite, not a claim
that all unrelated project tests or full-instrument scientific qualifications
were rerun. No commit or push was requested for this change.
