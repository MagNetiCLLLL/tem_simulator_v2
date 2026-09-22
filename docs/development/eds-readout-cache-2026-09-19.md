# EDS dose and spectral readout from executed response

This continues the [persistent EDS archive work](eds-persistent-cache-2026-09-19.md).
The current scope is classical particles. Coherent development remains paused.

## Implemented behavior

After an EDS calculation at a positive physical dose, retain compact immutable
float64 expected-response coefficients per electron reaching the specimen
reference plane. These coefficients come from the executed material tracks,
ionisation and photon transport. They are not a replacement electron source.
The existing full vacancy, line, photon, material and identity records remain
available for inspection and persistence.

For the same normalized incident distribution, material and collection geometry:

    new expected response = retained rate per arriving electron * new arrivals

The separately signed `eds_response` product excludes only column current
percentage, spectral maximum/bin width/FWHM and Poisson enable/seed. When both
scan controls are off, changing frame period changes default dwell but cannot
change static electron trajectories, so that period may also reuse the response.
Source emission, incident angles/energies/positions, material state, numerical
sampling, detector efficiency and geometry remain dependencies. Active scanning
retains its timing dependencies. No angular or spatial bucketing was introduced.

- Dose changes scale vacancy, line, photon-path and per-detector expectations
  consistently. Physical probability weights and geometry remain unchanged.
- Spectral range/bin/FWHM changes re-form the spectrum from all retained physical
  lines, including lines outside the previous plotting range.
- Noise-only changes retain the exact expected spectrum and draw new Poisson
  counts. Previously sampled noisy counts are never multiplied by a dose ratio.
- A positive-dose response can replay to zero and back to a positive dose. An
  initial zero-dose or historical result without coefficients falls back to the
  physical calculation when a changed readout requires a response.
- Both ordinary full calculations and saved-section continuation participate.
  The result reports `eds_response` reused and `eds` calculated, and the timing
  panel displays "EDS dose and spectral readout" for the actual work.
- Acquisition source/arrival counts and event/conservation ledgers correspond to
  the new result. Reused acquisitions avoid rebuilding the same ledger twice.

The archive admits the new coefficient type through its explicit numeric/JSON
decoder and validates its shapes, finite nonnegative values, row identities and
detector segments against the saved physical records. Older archives without
coefficients remain readable. A solver-source change still invalidates active
calculation reuse; historical files are not relabelled to bypass that check.

## Verification

Validation completed on **2026-09-20**. All **434 related
regression cases passed**, with no failures, errors or skips (366.464 s).
This includes response physics, cache invalidation, archive/controller/UI,
event ledgers, source-versioned native caches and a real nine-particle
tip-origin archive/continuation case. See the
[machine-readable receipt](eds-readout-cache-validation-2026-09-19.json) for the
exact test-file selection and implementation identity.

After this broad run and the timing pair, a final **progress-label-only**
correction ensures a missing saved EDS product reports full calculation rather
than response updating. Its 6 focused fallback/reuse cases
passed (45.411 s). The receipt keeps the measured and
final implementation hashes separate. No numerical/cache-admission/transport
code was changed by that last correction, and no old archive was rebound to a
new implementation identity. The two selections overlap and are not added as
independent test totals.

Serial `compileall` over source, tests and scripts and the whitespace/diff
check both passed after the tests exited. The running user application was
not restarted; native GUI interaction beyond the offscreen tests was not
claimed. No commit or push was performed.

A separate four-case supplied-material-track comparison against the prior
EDS implementation preserved cold-calculation bins, expected/sampled counts,
vacancy/line records and photon transport exactly. It is a material/EDS check,
not an independently validated full electron gun.

### Actual 5,000-particle loaded-archive comparison

One 300 kV classical CPU run, flat tip, 5 nm finite Si reference specimen with
100 micrometre diameter, all 5,000 trajectories hitting material. Column step
0.5 mm, gun step 0.2 mm, both histories 2 mm; EDS enabled, filter/wave/scan/vacuum
disabled. A true tip-origin section at Z = 1589.2 mm was saved/restored and
continued through the specimen to the full endpoint. The saved full EDS
product contained 82,592 line records and survived an exact
full-precision archive comparison, including its response coefficients.

The controlled pair loads that same executed state, halves only the column
current percentage, and compares response replay with recomputing EDS while
retaining identical material histories. Both use 12 numerical threads and
serial BLAS. The simultaneous regression worker was capped at 4 threads;
combined numerical budgets did not exceed 16 of this host's 32 logical CPUs.

| Current calculation | Recompute EDS | Reuse response |
| --- | ---: | ---: |
| EDS stage | 99.627 s | 1.086 s |
| Whole calculation call | 110.294 s | 11.777 s |
| Save completed result | 39.868 s | 39.073 s |

Whole-call time fell **89.3%** in this one pair.
The archive load took 45.329 s and was shared by both branches.
For an indicative load + calculation + save comparison, applying that same
measured load to each branch gives 195.491 / 96.178 s
(recompute / reuse). This is not two independent end-to-end IO runs.

Replay executed **zero** gun, material or EDS-physics calls. The control made
exactly one EDS call and zero gun/material calls. All physical EDS records,
event ledgers, conservation and coupling records agreed at rtol 2e-12,
atol 1e-12; the largest absolute expected-bin difference was
5.45697e-12. Original material/source identities
were retained. This is one controlled pair, not a timing distribution or
complete microscope qualification.

Additional in-memory whole-call measurements were:

- Noise-only redraw: 1.428 s; exact expected-array reuse and the specified Poisson seed verified.
- Bin width/FWHM change: 1.580 s; identical executed photon transport retained.
- Dwell doubled with both scans off: 6.523 s; expected counts doubled without retracing physics.

The initial pre-specimen section took 61.368 s,
and its continuation took 135.171 s.
Initial complete-result save/load took
37.519/46.839 s.
These separately measured setup/IO costs are not included in the replay
speedup. Generated archives and numeric arrays remain local and ignored.

### Compiled-code cache regression found during testing

The existing native compilation cache caused a repeatable Windows access
violation in the nine-particle completed-section fixture. Identical inputs
passed with a fresh cache and on a subsequent warm reload. The old cache was
retained for diagnosis. This isolates a persisted compiled-artifact problem;
it does not establish exactly which old artifact became inconsistent.

The application now selects a source-version-specific compilation cache before
importing Numba. Its identity includes all installed simulator Python source,
so a change in a dependency cannot admit machine code from a prior source tree.
Explicit `NUMBA_CACHE_DIR` settings are respected, and an already imported
Numba runtime is not reconfigured. The first calculation after a code change
may therefore compile again; later runs of the same version can reuse it.
Numba documents a cross-file dependency invalidation limitation in its
[compilation cache guidance](https://numba.readthedocs.io/en/stable/user/jit.html#cache).

In this environment a pytest plugin imports Numba before importing the
simulator. The final regression therefore bootstraps `import temsim` before
`pytest.main(...)`, matching application startup without forcing a cache path.
Plain `python -m pytest` does not migrate an already active Numba configuration;
that late-startup limitation remains explicit and is tested.

The full-size archive check also caught an ordering error in the new response
validator: JSON object keys may be sorted on disk. Alignment now compares key
sets for mappings while preserving the separately stored ordered key tuples
that index coefficient arrays. A multiple-emission disk fixture verifies both
unchanged records and correct per-key scaling after this reordering.

## Limits and next work

This improves repeated acquisitions and live readout changes. It does not skip
the first material/ionisation/photon calculation, and it does not make a changed
incident direction equivalent to a count-only change. Full particle archive IO
can still be substantial because the complete executed state is retained.

Custom low-level point requests with explicit dose/dwell or custom geometry are
conservatively checked by the existing request guard. They are not silently
accepted as a default UI acquisition. These changes do not establish coherent,
active scan, energy-filter or complete microscope qualification.

Next, profile first-run EDS substages and avoid repeated exact photon
intersections/attenuation and unnecessary intermediate record construction,
while retaining source/line identities and per-detector accounting.
For repeated live acquisitions, also profile archive encoding/decoding and
consider sharing unchanged immutable records between saves, without dropping
executed states, precision, integrity checks or explicit calculation endpoints.
