# First qualitative-science batch

## Scope and implementation

This batch covers a reproducible classical software baseline, primary tip/gun/
lens/aperture parameter meanings, and bounded particle trends. It does not
qualify the complete microscope, coherent imaging, a material-dependent lens
field, live hardware, GPU performance or the energy filter.

1. Component-operation tests now copy the complete configuration catalog and
   linked definitions to their own temporary directory. Saved composed modules
   are compared after resolving their subassemblies. Original source and
   dependency bytes are checked unchanged. A second occurrence in the parameter
   panel fixture was reproduced and fixed the same way.
   Later regression also reproduced seven pre-existing failures in the archived
   preceding implementation: voltage-reference and support-probe tests assumed
   a surface emitter was still the default; the C1 test assumed a divergent exit
   beam. Surface-only checks now select that model explicitly. The C1 check
   evaluates the actual aperture crossing for either a converging or diverging
   beam, retaining the distinct-plane and stopped-particle assertions.
2. The existing semantics/registry now reports missing or previously truncated
   units, including nA, T, Pa, T/m, V/m, V/m² and inverse-length powers. Primary
   tip/gun controls include their mechanism and held-condition limits. Historical
   emitter metadata is distinguished from active controls. Existing parameter
   panels and the source editor reuse these explanations.
3. An endpoint-energy regression found and corrected a real transport handoff
   error. With an analytic extractor transition from -1 to 6 mm, the launch
   potential is nonzero. The executed gun trajectory accounted for it, but the
   downstream `energy_offset_ev` copied the original launch offset. That made
   the downstream energy approximately **93.05646457 eV too high** in this
   diagnostic setting. The handoff now includes the actual launch-to-exit
   potential correction, with the historical nominal-energy convention retained.
4. `ANALYTIC_ENERGY_SCHEMA` changes to `launch-potential-handoff-v2`. The existing
   gun and downstream cache identities consume this schema, so a corrected run
   cannot admit an earlier incompatible result. Historical files are not
   rewritten and no new downstream source is introduced.

The parameter-to-implementation inventory and model limits are recorded in
[the parameter inventory](qualitative-parameter-inventory-2026-09-18.md).

## Scientific checks

All source-to-lens checks launch at the actual tip and execute extraction,
acceleration and gun apertures. Vacuum participation is off. The closed-aperture
case must transmit zero; finite aperture cases must include nonzero and partially
transmitted populations, so a completely lost beam cannot pass vacuously.

| Check | Conditions and criterion |
| --- | --- |
| Prescribed current | 49 and 193 particles; hold geometry, energies, directions and fields fixed. Double current; identical trajectories and exactly doubled weighted gun flux within relative 1e-13. |
| Nested gun apertures | C1 radius 0, 0.0005, 0.001, 0.002, 0.004 and 0.01 mm; fixed centre and launch samples. Surviving sets must be nested, with current equal to emitted current times original surviving weights. No renormalization. |
| Energy handoff | 49 particles; extractor transition starts at 0.05 or -1 mm, ends at 6 mm. Compare downstream energy with launch kinetic energy plus the actual endpoint potential difference; absolute tolerance 1e-8 eV. This is bookkeeping consistency, not an independent accuracy certificate for the entire integrator. |
| Isolated field response | Installed first condenser lens in ideal-field mode; 1%, 2%, 4% excitation. Signed field reverses with polarity; integrals of B² have ratios 1:4:16. Other overlapping fields and material saturation are outside this isolated limit. |
| Tip-origin first-lens segment | Actual gun output to 550 mm; installed first lens at 2%. Compare opposite polarities and independently refine gun and column integration. Finite source sampling is retained; this does not qualify specimen illumination or full images. |
| Cache boundary | Previous energy schema produces distinct gun keys and invalidates dependent incident, column and image/spectroscopy products. |

No physical default, source geometry, aperture opening, lens strength, dependency
version or source-file record is changed to make a diagnostic pass. Test-only
operating values exist in separate states. Coherent development remains paused.

## Validation receipt

Classical software acceptance: **259 passed**, exit 0. Its source/configuration
hash inventory was unchanged during the run. The receipt deliberately reports
`full_simulator_qualification: UNQUALIFIED`. This run completed before a test-only
refinement switched the new column diagnostic from float32 display histories to
float64 execution checkpoints; production code did not change afterward.

```powershell
.venv/Scripts/python.exe scripts/validate_classical_scope.py --scope classical --output tmp/qualitative-20260918/classical-baseline --timeout-seconds 900
```

The focused regression command is:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$checks = @(
    'tests/test_qualitative_particle_trends.py',
    'tests/test_qualitative_parameters.py',
    'tests/test_parameter_semantics.py',
    'tests/test_parameter_semantics_ui.py',
    'tests/test_component_operations.py',
    'tests/test_component_persistence.py',
    'tests/test_magnetic_lens_aberration.py',
    'tests/test_gun_aperture_planes.py',
    'tests/test_gun_adaptive_step.py',
    'tests/test_gun_voltage_reference.py',
    'tests/test_analytic_gun_field.py',
    'tests/test_gun_zero_field_evaluation.py',
    'tests/test_tip_preview_transport.py',
    'tests/test_tip_assembly_particles.py',
    'tests/test_continuous_tip_curvature.py',
    'tests/test_lens_field_provider.py'
)
.venv/Scripts/python.exe -m pytest @checks -q -o addopts= -o junit_family=xunit1 --tb=short --junitxml=tmp/qualitative-20260918/verified-regression.xml
```

The first focused run collected **213 cases: 206 passed, 7 failed**. All nine
new physical-trend cases and all fourteen new parameter cases passed. The seven
failures were the stale fixture assumptions described above; production code
did not change after this run. The complete three affected test files are rerun
with:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_gun_aperture_planes.py tests/test_gun_voltage_reference.py tests/test_tip_preview_transport.py -q -o addopts= --tb=short --junitxml=tmp/qualitative-20260918/fixture-followup.xml
```

Measured bounded results from the focused run:

- Doubling prescribed current gives a transmitted-current ratio of **2.0** for
  both source sample counts, with identical particle paths.
- The 0.0005 mm C1 opening transmits approximately **8.980 µA** with 49 particles
  and **8.912 µA** with 193 particles; 0 mm transmits zero and the tested openings
  of 0.001 mm and above transmit 10 µA. These values describe the declared
  diagnostic setting, not measured microscope performance or certified sampling
  convergence.
- Maximum endpoint energy-bookkeeping discrepancy after repair is
  **5.82e-11 eV**, versus **93.05646457 eV** before repair in the nonzero-launch-
  potential case.
- Installed-lens field-squared integrals have the relative values **1, 4, 16**.
- Halving gun integration/drift steps changes exit positions by at most
  **2.496 nm**, below the predeclared 10 nm local budget. Column checks use
  **251, 501, 1001 nodes**, with float64 endpoint differences of approximately
  **1.56e-17 m** and **7.06e-19 m**. These are consistency results for a smooth,
  weak-field segment, not physical accuracy or resolution specifications.

Affected-file follow-up: **25 passed**, exit 0, including all seven previously
failing cases. With these complete file reruns replacing their earlier outcomes,
all **213 selected cases have passing evidence**. This is an initial run plus
scoped reruns, not a claimed second uninterrupted 213-case run. Counts overlap
with other receipts and must not be added as unique tests.

Additional source-admission/editor/default-source regression: **59 passed**.
Python compilation and `git diff --check` passed after the final edits. Existing
Pydantic deprecation warnings remain; the full repository suite and native
desktop were not run. No long coherent calculation, hardware acquisition,
commit, push or application restart was performed.

The corrected full-column support-probe test explicitly selects the classical
surface emitter. Its axis probe has zero current: this proves path/interception
semantics, not nonzero sample illumination or an image-quality acceptance.

Raw diagnostic outputs are retained locally under `tmp/qualitative-20260918`;
generated particle arrays and numerical caches are not delivery artifacts.

## Next work

Continue with the remaining parameter inventory and coupled stigmator/corrector/
scan responses, followed by detector/loss presentation and bounded working-point
sweeps. Integrate the energy filter last. Retain source-specific model limits and
independent numerical checks instead of fitting commercial calibration tables.
