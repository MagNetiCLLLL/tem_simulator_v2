# Project audit fixes — 2026-09-21

Scope: the five reproduced functional defects and packaging contamination in
[the read-only audit](project-audit-2026-09-21.md). Fixes preserve the existing
classical particle model, current archive schema and tip-origin calculation
chain. Coherent development remains paused. No application restart, Git commit
or push was performed.

Status: completed. **5,926 passed, one skipped, zero failures and zero
unexecuted cases**. All four final test processes exited successfully.

## Corrected behavior

1. **Physical filter-plane population.** Source-position, emission-angle and TOF
   tracking, plus advanced transverse observables, share the same executed
   physical crossing. The cache also identifies the selected component, and
   outgoing coordinates use the filter's local frame. An arbitrary downstream
   global Z without a recorded crossing is unavailable. Missing clock data does
   not discard valid geometry. Source identities, path ancestry, interaction
   categories and absolute particle weights are retained.
2. **Archive identity.** Queued same-path replacements invalidate old bindings,
   including hard-link aliases. Fast reuse checks a previously verified manifest
   receipt and stable file fingerprint. Previously unknown automatic archives
   receive bounded streaming array-checksum verification in the file worker;
   no second large result is decoded. Delayed save/load completions are rejected
   if the file changed. Corruption is reported explicitly, never as saved success.
3. **Cancellation.** Ordinary calculation and STEM-frame entry points consume
   the cancellation token at existing sample-history, EDS track/photon-batch and
   STEM-row progress boundaries, including callers without a display callback.
   The worker checks before and after progress signals. Cancelled work cannot
   publish or persist an incomplete replacement result. An active numerical
   kernel exits at its next existing safe boundary rather than being interrupted.
4. **Camera pixels.** The shared runtime/profile validator requires a positive
   integer. Invalid edits restore the field and do not publish changes; invalid
   profile application leaves all controls unchanged. Valid counts survive full
   state cloning.
5. **Experiments workspace.** Restoring a layout explicitly schedules the
   existing read-only Design Explorer status refresh after blocked tab signals.
   It runs only when visible and dirty, without requesting a calculation or
   changing physical inputs.
6. **Wheel contents.** Wheel builds allocate their own fresh temporary build
   and staging directories. Prior build output is neither consumed nor deleted.
   Skipping the source build is rejected. No legacy adapter was added.

## Reproductions and focused verification

| Area | Before fix | After fix |
| --- | --- | --- |
| Filter display | Same named plane showed 2 arrivals / 30% source for TOF, but 16 / 100% for position and angle | All three use 2 arrivals / 30%; 175 focused cases pass |
| Archive | Executed 49-particle sections A and B overwrite one file; A falsely reports reuse after B | 68 related cases pass, including overwrite, deletion, corruption, pending replacement and delayed completion |
| Cancellation | Cancelling at sample history 0/32 still completed 32/32 | 107 related cases pass; bounded ordinary sample/EDS calls exit, and a real 3×2 geometric STEM frame stops after the first row |
| Pixels and workspace | Six invalid pixel paths and stale Experiments status fail | 30 related cases pass |
| Packaging | Seeded retired Python modules enter the wheel | Current-module byte comparison and stale-module/resource exclusion pass |

Filter display uses a deterministic saved-data fixture, not new physical filter
qualification. Archive regressions execute real small particle sections.
Cancellation fixtures isolate ordinary application entry points and real local
material/EDS/STEM kernels; they are not full-instrument qualification.

Independent packaging review passed PEP 517 requirements, metadata, wheel,
sdist and wheel-from-sdist, with runtime package/dependency imports deliberately
blocked. A wheel built directly from the dirty checkout contains exactly 447
current Python modules with matching bytes and zero extra or missing modules.
The stale ten modules found in the audit are excluded. Installation to an
isolated target and offscreen startup create all 13 main tabs without transport;
the target uses its own configuration root, settings and cache directory.

## Full regression

Completed against frozen input digest
`69f650de1cd2dc8e7601332ef6aca028d0527101ce81cca202717f2250b2750b`.
The fresh collection contains 5,927 cases. Four isolated processes use two
numerical threads each and one thread per nested BLAS pool, within the user's
half-logical-CPU limit. Live application settings and calculation caches are
isolated from this verification.

| Group | Passed | Skipped | Seconds |
| --- | ---: | ---: | ---: |
| 0 | 1,752 | 0 | 900.582 |
| 1 | 566 | 1 | 911.471 |
| 2 | 1,805 | 0 | 1,096.190 |
| 3 | 1,803 | 0 | 1,169.512 |

The only skip is
`test_fixed_lens_aperture_scaling_matches_recalculated_toml_metrics`: its stored
reference predates the C2 aperture relocation. No case was excluded from the
fresh collection. Final checks found no changed input among the 1,056 files in
the frozen snapshot. Serial `compileall`, `pip check` and `git diff --check`
passed. These are regression and contract checks, not qualification of absolute
performance or the full coherent microscope chain.

The initial run found three outdated test expectations in two modules: exact
progress-event lists omitted newly connected real sample progress, and a
timing-only fake writer omitted the required digest receipt. Those explicit
fixtures were updated, all 35 cases in the two modules passed, and the entire
affected shard was restarted from its first test. No production input changed;
the other shards do not consume those test modules. The initial attempt remains
in `shard-1-first-attempt` for inspection.

Receipts are local in `tmp/audit-fixes-20260921/`. Earlier failing reproductions
and focused receipts are retained alongside the audit and agent validation
directories. Generated archives, wheels and numerical outputs remain untracked.
The lightweight [verification summary](project-audit-fixes-2026-09-21.json)
records the final counts, input digest, affected files and packaging checks.
