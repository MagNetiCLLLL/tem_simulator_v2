# Tip-origin electron source and computed checkpoints

Status: the physical particle gun, an explicit tip mutual-intensity model and
a development variable-energy quadratic gun wave solver are available.
The stationary development path now also includes analytic column propagation,
finite-specimen zero-loss interaction and optional per-mode detector phase.
Complete general-field tip-to-image TEM/STEM propagation is not implemented.
The user withdrew the independent effective exit-source design on 2026-09-11.
See [tip parameters, equations and implementation status](TIP_TO_IMAGE_WAVE.md).

## Physical boundary

In this project the FEG gun contains the emitting tip, extraction electrode,
electrostatic gun lens, accelerator (including the DPA aperture), gun deflector,
gun stigmator and C1 aperture. The monochromated variant also includes its Wien
monochromator. The C1 condenser lens belongs to the downstream column.

The ordinary FEG configuration places the C1 aperture centre at gun-local
445 mm and the exit at 450 mm. These are simulator geometry values, not OEM
measurements or a universal microscope boundary. A filament/tip is only the
emitting part; it is not the complete gun.

## Customizable emission

Open **Model Inspector → FEG tip emission…**. This edits the existing emitter's
current, spatial width, angular distribution and launch-energy distribution.
All values refer to the tip launch plane, including the energy spread. Apply
changes the tip parameters; Cancel and invalid input leave the instrument
unchanged. The downstream energy is obtained through the accelerating fields.

An explicit Gaussian-Schell tip option adds incoherent angular spread,
wavefront curvature, emission centre and mean transverse momentum. It also
selects the same tip distribution's Wigner samples for particle diagnostics.
The old truncated classical emission law remains available when it is off.
This option supplies no exit energy, exit current or downstream source plane.

Extraction, acceleration, focusing, deflection, stigmation, physical apertures
and an installed monochromator remain part of the physical particle path.
Gun trajectories, DPA/C1 transmitted currents and arrival-time history are
preserved. Existing thermionic-gun controls and transport also remain available.

The old Gaussian–Schell dialog described a high-energy ensemble at the gun
exit and bypassed those upstream operations. This is prohibited even when its
parameters are bound to a digest of the gun. It is no longer an active model,
and the application does not offer a rebind/calibrate shortcut.

## Equivalent states mean cached computation only

A downstream state may be reused only when it came from actual upstream
transport and the relevant inputs match. The required dependencies include:

- Tip emission distribution and its current/energy reference.
- All consumed upstream geometry, fields, voltages, lens controls, apertures,
  deflectors and material/model inputs.
- Propagation boundaries, numerical settings and solver implementation.

An upstream change invalidates the affected result. Unchanged stages may be
reused to avoid repeating their computation; this never grants permission to
specify their output independently. The existing physical gun trajectory cache
continues to execute `trace_feg_to_exit` on a cache miss.

Historical exit-source configuration data and numeric wave arrays remain
readable as evidence. Activating an exit-source profile, restoring it into the
live instrument, generating a new exit beam, or using an old exit-wave cache
for a new physical calculation is rejected. Profiles are not silently converted.

## Remaining coherent-wave work

A tip-origin Gaussian-Schell mutual intensity and corresponding Wigner particle
samples are now implemented. The new `tip_gun_wave` development operator
executes extraction, acceleration, focusing, deflectors, stigmation, analytic
Wien fields and physical gun masks with variable longitudinal momentum.
Its scalar quadratic paraxial domain must not be confused with the original
Lorentz solver's general-angle dynamics or arbitrary imported field providers.

After that, the column, finite specimen entrance, specimen scattering, scan
coils and downstream detector/image adapters must consume the same derived
state. Independent phase, energy/current conservation, aperture-loss and
numerical-convergence checks are required. Existing specimen and propagation
kernels remain useful, but are not proof that this full chain exists.

New production wave images therefore remain unavailable. `--gun-wave-only`
does not generate an independent exit-source replacement. The separate
`scripts/trace_tip_wave.py` exports development tip-to-exit evidence from an
explicit tip-coherence profile; its archives do not qualify image admission.

`test_effective_gun_source.py` retains historical source-math checks only;
`test_gun_wave_transport.py` explicitly injects an isolated historical fixture
to retain column mathematics and codec tests. Production source-policy tests
run without that override. No isolated fixture qualifies tip-to-image acceptance.

See [the acceptance ledger](HANDOFF_V2_ACCEPTANCE.md) and the project rules in
[AGENTS.md](../../AGENTS.md).
