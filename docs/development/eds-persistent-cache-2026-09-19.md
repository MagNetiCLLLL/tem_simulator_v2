# Persistent exact EDS results in particle archives

Implementation and bounded acceptance recorded 2026-09-19. This implements
phase 1 of [the EDS cache plan](eds-cache-plan-2026-09-19.md): exact executed
products, with unchanged material/photon physics and numerical settings.
Coherent development remains paused.

## Behaviour

Ordinary full particle calculations and explicit cutoff-plane calculations
retain the complete existing EDS product in `MaterialSectionCache`. Archives
preserve the spectrum arrays, optional sampled counts, line/vacancy records,
material quadrature and retained photon transport records. This is the
executed product, including its existing configured storage limits; no new
photon path or physical interaction is inferred from a display curve.

Both calculation routes can reuse it after loading and editing an eligible
downstream projection lens. The existing executed gun, material histories and
column-prefix checks still apply. The interface's reused-product list records
EDS reuse, and full-pipeline progress explicitly labels restored EDS. Final
completion is emitted after checkpoint capture and external-input validation,
whose time now belongs to the final diagnostics stage.

Reuse requires the material/model signature, actual incident-array digest,
actual acquisition point and independent EDS signature to match. In addition,
the actual point call records its full detector geometry, explicit dose/dwell
overrides, custom surface/holder participation and numerical photon options.
Only the standard pipeline request for the matching assembly is admitted.
Custom API spectra remain supported and readable; a matching state signature
alone cannot relabel them as a standard cached acquisition.

Dose, bin width, broadening, noise seed and sampling changes still recompute
EDS in this phase. Position/direction/energy/weight and relevant material or
collection changes invalidate reuse. The non-scanning material-restoration
guard remains; there is no universal spectrum shared across raster pixels.

## Archive integrity and compatibility

The existing non-executable JSON/numeric-array format, checksums, immutable
loaded arrays and atomic writes remain in use. The explicit class whitelist
now includes the nested EDS records. On loading, the spectrum's executed
elastic state must exactly match the material cache before their shared
object identity is restored. Photon directions are validated and retained
bit-for-bit rather than normalised a second time.

Historical archives without EDS remain readable with `eds_spectrum=None`.
Historical spectra without actual call provenance are readable but cannot
be reused by this new path. Archive summaries report whether an actual EDS
product is present. Old solver identities are never rewritten to make them
compatible; a fresh calculation is required after a relevant code change.

## Validation

The related regression completed **257 passed, zero failures/errors/skips**
in **232.123 s**. It includes exact archive contents and corruption rejection,
legacy missing EDS, independent call overrides, equal-population incident
position/angle/energy/weight/clock changes, dose/pose/support/detector changes,
both pipeline routes, progress, material continuation, controller autosave
and working-point integrity. The regression includes a real nine-tip-particle
test with positive specimen hits and EDS counts, followed by loaded downstream
continuation while all gun/material/EDS execution entry points are forbidden.
Short drift fixtures elsewhere are explicitly artificial source fixtures.
Syntax and diff-whitespace checks passed. The whole repository suite and a
native visible desktop session were not run.

A fresh physical 5,000-particle run used the current solver, flat classical
emission, 300 kV, 0.5 mm column steps, the existing 0.2 mm gun step and 2 mm
history settings, a finite 100 micrometre reference-Si disk of 5 nm thickness,
and EDS enabled. Wave/coherence, scan, vacuum attenuation and energy-filter
assembly were off. The benchmark-only specimen is larger than the normal
small specimen to ensure actual material hits; no user profile was changed.
All 5,000 numerical primary histories hit material, with 166 elastic events and 82,592 EDS line
records. Expected EDS total was 25,887.507241510004 counts. Source probability
was conserved and positive-loss event-depth clocks stayed explicitly unknown.

The benchmark used **12 numerical threads, BLAS 1**, on the 32-CPU host. The
parallel regression used at most 4 threads, keeping aggregate numerical work
at or below 16. Each benchmark process had a 600 s cancellation/watchdog bound.
No previous solver identity was overridden to admit an old cache.

The fresh pre-specimen section took 46.626 s. Its loaded continuation through
the complete material path took 142.195 s, including 108.309 s for EDS. The
result was saved and loaded, and its entire EDS product was compared exactly,
including records and arrays. Its archive size is 2,687,177,994 bytes; peak
process RSS for the full/save/load phase was 10.44 GB.

For the controlled comparison, both continuations loaded that same archive,
used the same inputs, and increased projector lens 2 excitation by 0.1
percentage point. The control discarded only the optional saved EDS product
in its detached runtime cache, forcing the original EDS method to execute.
It retained the same executed gun/material state; it did not change physics,
numerical precision or any archived file. The accelerated run forbade gun,
material and EDS recalculation. Both complete EDS products matched exactly.

| Operation | Forced EDS recomputation | Exact EDS reuse |
| --- | ---: | ---: |
| Downstream-edit calculation | 118.209 s | 13.339 s |
| Load complete archive | 42.788 s | 43.106 s |
| Save updated complete archive | 37.129 s | 36.051 s |
| Total including load, calculation, save and checks | 201.757 s | 93.831 s |

The **calculation-only reduction is 88.7% (8.86 times faster)** in this single
controlled pair. It is not a general first-run or full-application speedup.
Loading/saving remains substantial and is explicitly separate: the reused
run took 93.831 s including load, calculation, save and benchmark checks.
Including those costs, this measured pair took 53.5% less time. The new
larger-archive IO is included on both sides of this controlled comparison;
this is not a measurement against an old-format archive's smaller IO cost.
Both runs reused all 2,310 incident integration nodes and resumed eight
material branches at Z = 2299.2 mm through the final Z = 3026.4 mm plane.
The EDS progress slot on reuse is bookkeeping; material compatibility checks
and derived ledger reconstruction are included in the material-stage time.

Local receipts are under `tmp/eds-cache-20260919`: `regression.xml` and
`material-acceptance-5000/acceptance-full.json`, `acceptance-reuse.json`,
`acceptance-recompute.json`, plus the bounded acceptance script. Generated
archives and numerical arrays stay local and ignored. No app restart, Git
commit or push was performed.
The lightweight [validation summary](eds-persistent-cache-validation-2026-09-19.json)
retains settings, identities, counts and measurements without numerical arrays.
The measured solver identity still matched the final installed source after
all runs; later changes in this turn were documentation and receipts only.

## Limits and next work

This change removes repeated EDS work; it does not accelerate a first EDS
acquisition or introduce a dose-normalised response cache. The next phase is
to separate expected response, physical dose, spectral response and Poisson
sampling, preserving all record weights and event accounting consistently.

The existing independent-atom and specimen-geometry limitations in the plan
remain. These tests qualify the declared classical cache/continuation scope,
not arbitrary tilted-specimen physics, coherent propagation, native desktop
interaction or complete microscope performance.
