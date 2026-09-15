# Particle transport recovery — 2026-09-14

## Scope

Recover tip-origin classical ray transport into the projection chamber without
removing extraction, acceleration, focusing, apertures, walls, or correctors.
Coherent-source development remains paused. This is an optical-transport
alignment, not a Nanoprobe focus, image, specimen-signal, or OEM calibration.

The installed tip, gun voltage reference, electrode positions, aperture sizes,
sample definition, downstream optics and detector insertion are unchanged by
the recovery operation. Only C1/C2/C3 excitation can be applied. In particular,
the earlier provisional tip-referenced near-gun arrangement is not silently
installed by this repair. Existing focus presets and saved profiles remain
unchanged; they can still be physically mismatched to a changed emitter.

## Reproduced cause

With the current factory particle state and 49 source samples, 35 rays exited
the gun. All 35 hit the column wall between Z=671.368 and 690.733 mm, before
the C2 aperture. Factory C1/C2/C3 were 90%, 35%, 55%. No current reached the
sample or projection chamber. The remaining rays stopped at the gun DPA
aperture (6) or gun-exit aperture (8).

Changing only C1/C2 could transmit all 35 through the existing 100 micrometre
C2 aperture, but the downstream column remained mismatched. A three-lens
proposal followed by the actual nonlinear trace was therefore necessary.

A detached Nanoprobe/Diffraction-preset comparison found a C1/C2/C3 candidate
at approximately 7.896755%, 25.749864%, 40.299464%. Twelve of 49 rays reached
the sample and passed the projection-chamber entrance aperture at
Z=2586.9 mm, carrying 25% of prescribed source current in this small sample.
Eight subsequently hit DF and four BF. These final detector stops are not
upstream transport failures. This exploratory small-sample value is not an
accurate transmission-efficiency calibration or a new focus preset.

## Completed forward checks

The production transaction was executed on the unchanged particle gun with
the Nanoprobe/Diffraction operating pair. It selected C1 = 8.012821602%,
C2 = 84.125639046%, C3 = 40.297740211%. These values are a recovered working
point for this assembly, not replacement factory defaults.

| Column mode | Positive-current rays at entrance | Fraction of source current | Radius at 0.1 mm step | Radius at 0.05 mm step |
| --- | ---: | ---: | ---: | ---: |
| Custom | 56 / 193 | 0.291666667 | 1.513656139 mm | 1.513661623 mm |
| Ideal | 55 / 193 | 0.286458333 | 1.471964836 mm | 1.471970558 mm |

Current and transmitted-ray count did not change when the column step was
halved in either mode. In Ideal, all 55 passing rays continued through
Z=2800 mm. The largest incident-column slope magnitude for these survivors,
after the executed gun exit, was about 70.17 mrad. Low-energy near-tip angles
belong to the separate gun integrator; they are not a column-angle diagnostic.

The real Ideal candidate also passed the GUI's immutable-state application
gate. An offscreen MainWindow showed the visible **Match transport** button
and the actual refined ray result; 16 of the display-selected rays passed
the projection entrance. GUI settings and artifacts were redirected to an
isolated temporary directory, leaving the running user's window untouched.
This is offscreen rendering evidence, not a live-desktop or image validation.

A subsequent ordinary **Preview** used the actual tuning preparation (49
samples, 1 mm column step, its zero-current tip diagnostic support). Eleven
positive-current rays still passed the entrance, with source fraction
0.229166667 and radius 1.463318348 mm. Refresh therefore did not empty the
chamber. The difference from 193-ray transmission is a sampling-budget
effect to qualify separately, not an established efficiency convergence.

## Application behaviour

Ray Diagram now has **Match transport**. It executes in the existing background
alignment transaction system and:

1. Captures the complete current instrument before searching.
2. Executes the physical gun with a bounded particle budget, reusing an exact
   upstream trace only when its consumed inputs match.
3. Proposes C1/C2/C3 strengths with first-order maps of the current column.
4. Rejects candidates that fail the normal nonlinear, physically clipped
   optical-only propagation.
5. Rechecks with 193 particles and 0.1/0.05 mm column steps. Both passes need
   at least four positive-current rays and 1% of source current past the
   installed projection-chamber entrance. Relative current and entrance
   envelope changes must each be at most 2%.
6. Atomically applies only those three raw controls, retaining the user's ray
   count, physics settings and other controls. A changed starting state,
   cancellation, or failed validation prevents application. Existing alignment
   undo/checkpoints remain available.
7. Displays the actual refined ray result as **Preview · transport validation**.
   It does not retrace for that display, promote it to High accuracy, calculate
   a specimen image, or erase completed image/spectrum results.

The operation does not run automatically on slider changes. Arbitrary lens
settings, an intentionally blanked source, closed pupils, or incompatible
geometry may legitimately have no transmitted beam. Failure remains explicit;
the matcher must not manufacture a visible ray to hide that state.

## Reproduction

Use the project interpreter:

```powershell
.\.venv\Scripts\python.exe scripts/check_column_matching.py --recover --preset --mode custom
.\.venv\Scripts\python.exe scripts/check_column_matching.py --recover --preset --mode ideal
```

The CLI prints lightweight measurements and verifies the same transaction
application used by the GUI. It writes no calculated arrays or cache files.
The historical exploratory search remains available without `--recover`.

## Validation boundaries

The step comparison qualifies column transport at the tested ray budget and
input state. It does not establish gun-field mesh convergence, arbitrary
source/geometry reachability, high-order nonparaxial column accuracy, or full
TEM/STEM imaging acceptance. Imported/coupled vector-field recovery is rejected
explicitly rather than replaced with an on-axis analytic search.

Focused transaction/readout/GUI regressions: 25 passed. These include physical
entrance-plane selection, permanent-aperture behaviour, exclusion of zero-
weight support rays and earlier stops, detector-hit interpretation, nonfinite
path rejection, exact three-control scope, cancellation, zero-current refusal,
existing alignment commit/undo/staleness, and the new button connection.
Changed Python files compile; whitespace checks pass. The full repository and
expensive image/wave suites were not rerun.
