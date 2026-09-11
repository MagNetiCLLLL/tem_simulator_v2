# Versioned effective electron-gun source

Status: partial HANDOFF v2 implementation. This is not a complete TEM/STEM
wave-image release or a calibrated microscope model.

## Select a model

Open **Model Inspector → Electron-gun source model…**.

- **Legacy classical particles** keeps the original emitter and its parameters.
- **Effective exit Gaussian-Schell v1** is a separate, versioned cold-FEG model.
  Its plane is the installed gun exit, not an independently movable illumination
  plane. Enter the current at that exit explicitly; it is not inferred from the
  old emission-current setting.

Apply binds the new parameters to the current gun. Changing gun geometry or
physical controls invalidates that binding and requires another explicit Apply.
Changing ray sampling or C2/C3 strengths does not recalibrate the gun. Cancel
does not change either model. Switching back to the legacy model retains the new
model's parameters on a separate shelf. Profile format 7 saves both; old profiles
remain legacy and are not silently converted.

## Parameter meaning

| Input | Reference and meaning |
| --- | --- |
| Exit reference current | Amperes at the gun exit, before column losses |
| Exit source FWHM | Gaussian spatial intensity FWHM at the gun exit, nm |
| Extra incoherent angular RMS | Additional per-axis angular RMS, mrad; adds in quadrature to the diffraction-limited spread |
| Exit energy FWHM | Gaussian energy spread, eV, around the gun's nominal exit energy |
| Energy nodes | Gauss-Hermite quadrature; convergence must be checked separately |
| X dispersion / angular dispersion | Correlated position/angle shifts per eV, at the same exit plane |
| Omitted mode weight limit | Upper bound on the omitted Gaussian-Schell eigenmode weight; retained weights are not renormalized |
| Mode budget / source grid pixels | Numerical resource and sampling limits, not physical controls |

This is a phenomenological equivalent source, not a resolved quantum calculation
of the emitter/accelerator. It has no measured calibration by default. Source
positions and canonical directions have a quantum-consistent covariance. The
particle column samples the same full positive Gaussian Wigner distribution;
coherent modes represent its explicitly truncated density operator. Energy
dispersion remains correlated rather than being discarded after propagation.
Gun-internal trajectories, DPA throughput and launch-to-exit time are unavailable
for this equivalent model and are not fabricated.

## Available computation and cache

Preview, Medium and particle High accuracy use the selected source. The coherent
producer transports gun modes through the shared quadratic paraxial field graph,
physical deflector actions and aperture planes to the finite specimen's upper
face (`centre Z - thickness / 2`). It does not propagate through half the
material as vacuum to reach its centre. The envelope reference stays fixed on
retraction; a thickness change moves this entrance and invalidates its cache.
Sampled finite vacuum bores are absorbing boundaries. Their axial sampling needs
an independent convergence check. Imported 3-D fields, unsupported nonlinear
phase terms and equivalent thin-lens actions are rejected explicitly.

The disk cache stores all complex mode arrays, affine grids, origins, quadratic
phase/tilt carriers, energy labels and surviving electron weights. It uses the
existing quota/checksum store. Complete-snapshot identity is separate from the
actual incident-operator signature. A specimen-content change can reuse the
incident modes when the entrance plane and all operator inputs are identical;
the new result records its new snapshot and original execution/parent links.
Changed fields (including overlapping downstream tails), apertures, entrance
plane or numerical sampling invalidate this incident cache. Warm requests
resolve dependencies but do not repeat coherent propagation.

With a saved profile selecting the new source:

```powershell
.\.venv\Scripts\python.exe scripts/cache_high_accuracy.py --profile SOURCE_PROFILE.toml --gun-wave-only --step-mm 0.2 --output outputs/gun-checkpoint-001
```

This writes `working-point.temwp`, the exact request manifest/profile and a
readback report. **Working Points → Import…** opens the package read-only.
Explicit Restore/Continue restores its compatible captured parameters without
presets or refocusing. To verify the disk product without repeating propagation:

```powershell
.\.venv\Scripts\python.exe scripts/cache_high_accuracy.py --verify-existing --output outputs/gun-checkpoint-001
```

Wave diagnostics read the saved numeric modes, not the active instrument.
`canonical_alpha95/99` are canonical Fourier-angle diagnostics. They do not
replace the mechanical-ray `alpha95` used by the existing Direct Alignment
target. In a magnetic field those bases differ. Unresolved phase sampling or
excessive diagnostic scratch space returns an explicit unavailable/out-of-range
status. It does not return a fabricated zero angle.

## Still unavailable

New production TEM/STEM images remain gated: their specimen, scan-coil and
post-specimen adapters have not yet been integrated with this coherent producer.
The old specimen-plane pupil is not used as a replacement. Existing images can
still be viewed; ray/particle calculations remain available with wave-image
products disabled. No source-grid, 32/40 mrad, specimen, CPU/GPU or full-domain
validation is implied by a successful cache readback.

Affine scalar phase and continuous metaplectic lift are now retained along the
quadratic field path. Complex analytic/FFT, composition/inverse and winding
tests cover a bounded domain; see `CANONICAL_PATH_PHASE.md`. Unresolved
mixed-conjugacy maps and complete-domain phase validation,
portable remapping of missing external model files, complete GUI/CLI workflow
acceptance and production-scale GPU/streaming evidence remain open. See
`HANDOFF_V2_ACCEPTANCE.md` for the current acceptance scope.

## Method references

- Starikov and Wolf (1982), coherent-mode representation of Gaussian-Schell
  model sources: https://doi.org/10.1364/JOSA.72.000923
- Collins (1970), ray-matrix diffraction integral:
  https://opg.optica.org/josa/abstract.cfm?uri=josa-60-9-1168
- abTEM, incoherent intensity averaging across source/thermal ensembles:
  https://abtem.readthedocs.io/en/latest/user_guide/tutorials/partial_coherence.html

These support numerical methods, not an independent specimen-illumination
production interface or an OEM source calibration.
