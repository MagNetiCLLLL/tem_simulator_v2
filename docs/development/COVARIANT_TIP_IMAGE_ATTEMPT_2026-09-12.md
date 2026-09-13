# Covariant boundary kernel and physical-tip image attempt

Status: **partial implementation; no image generated**.

## Requested outcome and actual result

The requested endpoint is an image from an executed physical-tip, gun,
column, specimen and detector chain, retaining the 5 nm spatial width,
0.3 eV mean emission energy and existing energy distribution. An isolated
boundary solution or an old preview is not that endpoint.

`scripts/attempt_default_tip_image.py` now records and attempts that chain
using the project's current default Si[110] specimen (5 nm thickness), the
explicit existing `TipCoherence()` selection and all original emitter values.
It requests the first inserted STEM detector, HAADF, before the energy filter.
It does not retract the screen, move a detector, change a lens or replace the
source. This is a single detector-plane attempt, **not a completed raster**.

Reproduction, with a new output directory each time:

```powershell
.\.venv\Scripts\python.exe scripts/attempt_default_tip_image.py --output <new-evidence-directory>
```

The run in `evidence/20260912-physical-tip-image-attempt-02/` failed at
`generate_tip_emission`, before a gun checkpoint was published. The unchanged
source-domain check reports `p_transverse/p bound = 2.78989`, above the
forward-support limit. Unlike the earlier monoenergetic compatibility fixture,
this request retains the actual nonzero energy spread. No wave, detector
image, or replacement-source cache was exported. `input_audit.json`, the
original profile and `result/failure.json` preserve the input and failure.

An earlier attempt in `evidence/20260912-physical-tip-image-attempt/` stopped
inside the diagnostic script on an invalid `Sample.preset_key` attribute.
It created only the input profile; it did not execute wave transport. That
script defect was corrected to read the existing `SpecimenScene`, and the
directory was preserved rather than overwritten or counted as a physics run.

## New numerical capability

`physics/covariant_boundary.py` solves

```
D = d/dz + i G
D^2 f + Q f = 0
G = e A_z / hbar  (electron charge is -e)
j_z = (hbar/m_e) Im(f* Df)
```

Both `G` (inverse metres) and `Q` (inverse square metres) are Hermitian
transverse matrices. They may be noncommuting. Each constant layer uses a
short matrix exponential in the state `(f, Df/kappa)` and scattering-matrix
doubling. Long exponentially growing transfer matrices are not formed.
Reflection embedding composes the layers and reconstructs both `f` and `Df`.
The returned derivative is explicitly **covariant**, not the ordinary `f'`.

The numerical chart `kappa` is not an injection energy. The left boundary
prescribes a total field, not an independent downstream source. The right
load has `G = 0` and outgoing/decaying derivative `i sqrt(Q_exit)`; arbitrary
nonzero exit connections need their own solved outgoing boundary load.

This is an internal scalar nonrelativistic operator. It is **not yet connected
to the installed gun's magnetic providers**, and does not supply transverse
vector potentials, apertures, absorbing material or an emission law. A caller
must assemble all terms consistently. In a truncated Galerkin basis, the
projected `A_z^2` is not generally the square of the projected `A_z`; the
corresponding variance term belongs in `Q`. This kernel does not silently
construct or omit that term. Its simple matrices are operator fixtures, not
the installed microscope field or full magnetic-gun acceptance.

The outgoing load supplies a positive spectral expression for current. If
the complex amplitude is representable but its squared current underflows,
the result retains the logarithm of that outgoing current and marks the
representation accordingly. If the coupled amplitude/transmission itself
underflows, execution fails explicitly; a general log-scaled matrix solver
is still needed. The interface conservation check is scaled to incident
numerical-chart current, not a relative precision claim for extremely small
transmitted current. No mode is clipped or renormalised to make a test pass.

## Validation

- `evidence/20260912T195128Z-covariant-initial-9b9f84d1/`: 14 passed,
  0 failed. Initial covariant and previous electric coupled kernels.
- `evidence/20260912T195617Z-covariant-validation-6de004bf/`: 67 passed,
  0 failed, 0 skipped, 33.14 s. Gauge phase versus unchanged current,
  noncommuting connections against an independent first-order exponential,
  basis covariance, evanescent transmission, budgets, cancellation, source
  guards, previous electric operators and development CLI checks.

The independent DOP853 comparison for a smoothly varying connection and
electric operator gives complex-amplitude errors `9.16e-5`, `2.29e-5`,
`5.72e-6`, `1.43e-6` for 16, 32, 64, 128 intervals. The predeclared `1e-5`
final tolerance is unchanged. These are offline CPU numerical checks, not
full-chain imaging validation or GPU performance evidence.

## Physical-model decision before complete integration

Live sampling confirms that the current analytic gun potential is exactly
zero from the tip through 0.05 mm. It is approximately 0.0234384 V at 0.1 mm
and 15.4003 V at 0.5 mm. This is the installed compact polynomial extractor
model, not a calculated field around the 100 nm-radius tip.

The project has a 5-degree tip cone parameter, schematic/mechanical tip
dimensions, an extractor body and an extractor voltage. It does not have a
fully specified curved-tip electrostatic boundary problem or a calibrated
near-tip field map. Nor does its current paraxial Gaussian-Schell emission
law define the general nonparaxial current/near-field boundary and its
coupling to reflected/evanescent components. The new mathematical kernel
does not establish those missing physical definitions by itself.

Proposed next physical model, requiring an explicit decision: an opt-in,
versioned, idealised axisymmetric tip/extractor field and emission-boundary
model, derived from declared geometry and voltages and clearly marked as
uncalibrated. Existing source values, components and historical profiles
would remain unchanged; no automatic migration or downstream source would
be added. Its electrical outer boundaries and the meaning of the emission
flux must be documented, not fitted to obtain a desired image.

Choosing that model is only a starting point. The implementation still needs
its executed electric/magnetic transport, physical apertures/absorption,
matching to the existing column, source-dependent scanning/inelastic waves,
and image convergence. Until then the requested image remains unavailable.
