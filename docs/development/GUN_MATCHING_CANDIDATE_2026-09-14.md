# Gun-to-condenser matching candidate

## Status

**Experimental particle candidate, not an installed or qualified preset.**
The source/gun and original strong-objective branch are retained. The candidate
can form a small geometrical probe, but sampling, gun topology and full-column
acceptance must not be inferred from that spot. No defaults, open-window state,
emitting cap, angular/energy law, aperture opening or voltage reference was
changed. No coherent propagation or image calculation was run.

## Implemented matching support

- `gun_matching.axis_variational_map` proposes geometry from the *executed*
  scalar electrostatic field. It tracks both independent position/momentum
  responses, not an invented gun-exit beam. Coordinates are `(x/R, px/p_tip)`;
  integration uses a logarithmic axial coordinate to resolve the tip and the
  complete acceleration path. Its canonical determinant is reported.
- `candidate_with_gun_geometry` creates a detached candidate. Electrode bodies,
  optical references, module parts, vacuum bores and liners move together.
  Body overlap, changed component order and nonfinite placements are rejected.
  Source parameters, voltages, references and other parts survive a snapshot
  round trip. The executed-field request changes when the geometry changes.
- `search_gun_matching.py` screens only valid physical placements. It does not
  imply finite-cap or aperture acceptance. `check_gun_matching_candidate.py`
  performs ordinary tip-origin particle propagation and writes scalar evidence.
  Existing reports cannot be overwritten by these scripts.

The local equations are `dx/dz = px/pz` and `dpx/dz = -e Er/vz`, with the
relativistic kinetic energy determined by the same scalar potential. This
retains the acceleration damping of slopes; a constant-energy optical matrix
would not suffice. The momentum-coordinate formulation and electrostatic
paraxial equations are discussed in the [CERN Accelerator School lecture on
electrostatic lenses](https://cas.web.cern.ch/sites/default/files/lectures/zeegse-2005/ionoptics-final.pdf).
This reference is not an OEM calibration or finite-source validation.

## Candidate input settings

The reference assembly uses a 300 kV Nanoprobe / Diffraction operating state.
The tip remains R=100 nm, emitting-cap half-angle 10 degrees, with the same
normal/tangential emission law and 100 nA source current. Only numerical cap
quadrature is changed to `apex_stratified_v1`: all nine area strata and their
original current fractions remain, including rays later lost at real stops.

| Setting | Candidate |
| --- | ---: |
| Extractor centre from tip | 8 mm |
| Extractor control | 4.5 kV relative to tip |
| Gun lens centre from tip | 22.817987616230624 mm |
| Gun lens control | 1.1 kV relative to extractor (unchanged convention) |
| C1 excitation | 30.68886068895488% |
| C2 excitation | 15% |
| C3 excitation | 24.45025586236745% |
| Objective excitation | 68.9668378549914% |

The gun-lens body is 18.8179876--26.8179876 mm; it does not overlap the
accelerator body starting at 30 mm. Its dimensions and bore are unchanged.
The gun voltage convention remains provisional; it was not relabelled as an
OEM ground-referenced voltage. Vacuum-medium participation is off.

## Actual observations

With the new gun position but historical column excitations, 193 rays exit
the gun and none reach the specimen. Moving the gun alone is insufficient.
After the C1/C2 changes, rays reach the specimen with the original objective
branch retained. C3 and Objective then adjust the probe angle and focus.

At **769 emitted samples**, 40 reach the specimen; their weights sum to
`8.0463851e-5` of source current (8.0464 pA for the 100 nA source).
The current is not rescaled to the original source current after clipping.

| Column step | Sample d95 | Sample alpha95 | Waist relative to upper surface |
| --- | ---: | ---: | ---: |
| 0.05 mm | 0.623998 nm | 30.000000 mrad | approximately 0 nm |
| 0.025 mm | 0.623214 nm | 30.000000 mrad | -0.03039 nm |

These are classical particle diameters containing 95% of surviving current,
not a wave-optical resolution prediction. The effective weighted sample count
is only **8.742**, below the existing acceptance minimum of 16. The step check
is encouraging but is not sampling or gun-field convergence.

An independent field-only check confirms why the geometry must remain
experimental. At the *same* electrode positions and voltages, the dimensionless
angle-to-exit-position matrix coefficient B is 0.0425284 on the 8-cell electrode
mesh (16,000 variational intervals), and 0.0425289 with 32,000 intervals. Refining
the **field** to 16 cells per bore gives B=26.2131 at 16,000 intervals. The
canonical determinants are close to one (0.9999949 or better), but that does
not imply the focused geometry is field-converged. No new full particle
qualification on that refined field has been claimed, and the geometry was
not retuned between these field comparisons.

All-current gun-exit d95 changes from 1.521 mm at 193 samples to 2.403 mm at
769 samples. This additional sampling sensitivity must remain visible; do not
quote the coarser exit diameter as a converged gun design.

### Fixed-settings sampling comparison (completed)

The 1537-ray execution retained **identical** lens controls, electrode geometry,
source physics and numerical cap rule. It used the same 0.025 mm column step;
there was no retuning between the two following measurements.

| Quantity | 769 rays | 1537 rays |
| --- | ---: | ---: |
| Transmitted samples at specimen | 40 | 76 |
| Effective weighted samples | 8.742 | 7.374 |
| Fraction of emitted current | 8.0463851e-5 | 3.2857283e-4 |
| Current for 100 nA emission | 8.0464 pA | 32.8573 pA |
| Sample d95 | 0.623214 nm | 0.801235 nm |
| Sample alpha95 | 30.000000 mrad | 31.577628 mrad |
| Waist relative to upper surface | -0.03039 nm | +2.79025 nm |
| Whole gun-exit d95 | 2.40340 mm | 2.55249 mm |

**Sampling convergence fails.** More detected trajectories do not imply more
effective statistical support: newly sampled, much higher-weight rays dominate
the small accepted population. Do not renormalize that population, discard the
new rays, or retune and describe the result as a convergence comparison. The
1537-ray angle/focus also miss the existing 1% / 1 nm acceptance tolerances.
The 1537-ray audit took **486.57 s**, including fresh gun propagation and scalar
auditing; this is not a TEM/STEM image run. Its maximum gun-exit energy error is
5.82e-9 eV, but energy conservation alone does not qualify the trajectory/field.

### Crossover audit

Exact-plane refinement at 769 samples and a 0.025 mm column step gives:

| Intermediate crossover | Candidate Z (mm) | d95 at this waist |
| --- | ---: | ---: |
| C1 -> C2 | 509.434680899 | 29.2886 um |
| C2 -> C2 aperture | 708.843838641 | 5.01997 um |
| Probe DP22 -> Probe HPC | 1195.906763535 | 7.04781 um |
| Probe TL12 -> Condenser stigmator | 1372.451845572 | 9.03917 nm |
| Mini Condenser -> Objective | 1585.919240507 | 0.511169 nm |

These five intermediate waists have the historical order/component intervals.
The C1 figure includes all rays surviving there, not just rays later admitted
to the specimen. The first crossover has **not** recovered its historical
43 nm whole-bundle diameter.

The raw upstream audit also detects the deliberately focused specimen waist
at `1599.199997469558 mm`, 0.03044 nm before the upper surface. It therefore
reports six roots and **fails the unmodified strict five-root topology gate**.
No root has been hidden or gate weakened to promote this candidate. The
surface focus and intermediate transport topology need explicit separate
terminal-plane treatment before a complete qualification can be reported.

Gun topology is a separate unresolved discrepancy. The new 769-ray gun
history has one interpolated waist near 285.852 mm. An isolated execution
of the historical `0a71832` gun (193 rays, 1 mm diagnostic planes) produces
multiple candidate variance crossings near 30.36, 52.24, 114.09, 150.93,
189.03, 222.93, 262.03, 295.95, 339.03 and 374.00 mm. These gun-history
roots are **not exact-plane or step-qualified**. They cannot certify the
user's full crossover-invariance requirement, nor can the old GUI's single
selected gun-waist label substitute for a full comparison. Post-specimen
topology has not been tested in this matching run.

At 1537 samples the surface waist lies just downstream of the entrance, and
the raw incident audit contains five candidates at 509.415296, 708.854682,
1195.661873, 1372.452382 and 1585.919242 mm. Their ordered component intervals
match the historical incident reference. This pass at a fixed budget is not
a pass of gun topology, field convergence, sampling or the surface-focus gate.

## Reproduction

`outputs/crossover-audit-20260914/gun-matched-769.json` is the local scalar
report containing exact inputs, source model, field report and all roots.
It is not an active profile or a particle cache. Generated arrays/caches
remain excluded from Git.
Its elapsed time includes reuse of a gun executed earlier in this session;
it is not a cold tip-to-specimen performance benchmark.
`gun-matched-1537-fixed.json` in the same directory is the completed independent
sampling comparison with the unchanged candidate controls.

```powershell
.\.venv\Scripts\python.exe scripts/search_gun_matching.py --extractor-mm 8 --gap-mm 3 6 8.818 10 11 --intervals 8000
.\.venv\Scripts\python.exe scripts/check_gun_matching_candidate.py --lens-mm 22.817987616230624 --c1 30.68886068895488 --c2 15 --c3 24.45025586236745 --objective 68.9668378549914 --rays 769 --step-mm .025 --refine-crossovers --reference-report outputs/crossover-audit-20260914/historical-refined-769.json --output <new-report.json>
```

## Validation

66 affected tests passed across the completed runs, including analytic drift/relativistic acceleration,
focusing sign and fourth-order refinement; geometry and snapshot consistency;
tip sampling, electrode mesh, voltage reference, surface-focus and crossover
gates. These test fixtures are separate from the actual particle evidence.
No full-project suite or visible desktop interaction was run.
The final added cache-identity assertions required initializing the standalone
gun's declared vacuum context before the immutability baseline; both placement
cases pass after that test setup correction. The production placement does
not mutate the input state. Changed Python files compile successfully.

## Next bounded work

Resolve gun-field convergence and the gun-internal topology discrepancy first.
Improve quadrature of the small physically accepted tip phase-space region
without omitting outer emission or altering its physical weights. Then repeat
fixed-settings convergence before any final angle/focus adjustment or preset
promotion. Blindly increasing whole-cap samples is already too costly for the
statistical support obtained. No long image job is needed for these checks.
