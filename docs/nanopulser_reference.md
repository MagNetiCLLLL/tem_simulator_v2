# Standard specimen beam blanking and optional NanoPulser

## Location and source evidence

Ordinary specimen beam blanking is available in every simulated gun assembly.
The optional NanoPulser is an additional electrostatic device; choosing no
NanoPulser does not remove the ordinary blanker. Camera and EDS shutter or
acquisition controls belong to their detector systems. A detector shutter
downstream of the specimen does not, by itself, remove specimen illumination.

The [temscript COM interface documentation](https://temscript.readthedocs.io/latest/instrument.html#temscript.Illumination.BeamBlanked)
describes `Illumination.BeamBlanked` as driving the gun tilt coils strongly
enough to divert the beam before the condenser. Its `Gun.Tilt` control is
unavailable while this blanking function is active. This is primary
documentation for the Python wrapper's interface; it is not an OEM drawing
or a calibration of coil current to beam angle.

Thermo Fisher's [TEM Server 7.2 release notes, page 15](https://documents.thermofisher.com/TFS-Assets/MSD/Product-Information/TEM-Server-307413.pdf#page=15)
name a separate `GunTiltBlanker` alignment dialog for Titan and Talos, while
also recording removal of the `Gun Blanker` entry from the Talos shutter
monitor. These distinct software entries must not be interpreted as proof
that every Titan/Talos generation contains an identical dedicated blanker
assembly. The cited manufacturer material gives no proprietary coil
coordinates or drive-current calibration.

The [Thermo Scientific Iliad Ultra datasheet, DS0510](https://assets.thermofisher.com/TFS-Assets/MSD/Datasheets/iliad-ultra-datasheet-ds0510.pdf)
identifies an electrostatic NanoPulser between the electron gun and condenser
module. That supports the simulated location and deflection mechanism. It does
not supply the internal dimensions, electrode voltages or lens-current tables
used below. Its pulse specifications apply to the commercial product, not to
the static simulator.

## Standard blanker geometry

The standard model reuses the existing gun deflector/tilt coils. Their
manifest entries are annotated with `blanking_role = "standard_pre_specimen"`
and `blanking_implementation = "gun_tilt_coil_drive"`. This is a declared
model choice based on the documented mechanism; the shared implementation
across FEG, monochromated FEG and thermionic sources is not a claim that their
commercial hardware is identical.

| Gun manifest | Existing deflector key | Existing coil interaction planes | Existing gun-exit aperture plane |
| --- | --- | ---: | ---: |
| `FEG.toml` | `feg_deflector` | 402 / 418 mm | 445 mm |
| `FEG_Mono.toml` | `feg_deflector` | 452 / 468 mm | 495 mm |
| `Thermionic.toml` | `thermionic_deflector` | 402 / 418 mm | 445 mm |

These module-local coordinates are the simulator's existing mechanical
geometry, not newly measured or manufacturer-specified blanker dimensions.
Standard blanking adds no axial length, lens, optional module or aperture.
The ordinary alignment settings must remain available again after
unblanking; the additional blanking drive is a separate operating state.
The gun TOMLs declare a provisional 50 mT transverse magnetic field in the
upper coil while blanked. Existing gun propagation and bore/aperture
interception determine whether electrons are stopped. This value is a
simulation field parameter, not an OEM current or a measured current-to-field
calibration. The model does not infer coil inductance or switching time.

Ordinary blanking and NanoPulser blanking are independent specimen-illumination
controls. Opening one does not imply that the other is open. Condenser
calibration measures the illuminated state, so it temporarily opens both
controls and then restores the user's requested blanking states.

Use **Optical → Deflectors → Gun Deflector Pair → Blank beam** to operate the
standard blanker. The optional NanoPulser has its own **NanoPulser blanked
(static)** checkbox in its component parameters. Neither control is duplicated
on the main calculation toolbar.

## Declared NanoPulser model

`configs/instruments/beam_blanker/NanoPulser.toml` defines an optional module.
Selecting **None** adds neither an axial interval nor a NanoPulser clipping
aperture; the standard gun blanker remains present.
Selecting **NanoPulser** inserts this model after the gun:

| Quantity | Model value |
| --- | ---: |
| Additional module length | 80 mm |
| Electrode centre, relative to module entrance | 20 mm |
| Electrode length / plate separation | 10 mm / 1 mm |
| Limiting-aperture plane, relative to entrance | 60 mm |
| Limiting-aperture radius | 0.1 mm |
| Initial blanking potential difference | 500 V |

Every dimension and the 500 V operating default is an illustrative engineering
assumption, not an OEM NanoPulser specification. With the standard FEG gun,
the module entrance is Z = 450 mm, so the electrode and stop lie at Z = 470 mm
and 510 mm. The column and recording modules move together by 80 mm; the
sample moves from 1599.2 mm to 1679.2 mm. Distances within those downstream
modules stay unchanged. Other gun selections position the module relative to
their own exit plane.

The component is a transverse electrostatic deflector, not a round focusing
lens. The thin integrated kick follows the relativistic impulse relation

\[
\theta = \frac{e\,\Delta V\,L}{g\,p v}
= \frac{\Delta V\,(L/g)}{K(K+2E_0)/(K+E_0)},
\]

where the second expression uses kinetic energy \(K\) and electron rest energy
\(E_0\) in eV. A positive plate-potential difference deflects electrons toward
the positive coordinate axis; azimuth rotates that direction. The nominal
beam energy sets this kick. The model does not resolve the tiny change in
deflection caused by the sub-eV source energy spread.

**Open** sets the kick to zero while retaining the installed aperture.
**Blanked** applies the configured voltage and traces interception by the stop;
it does not delete rays without a physical interception. Consequently zero or
insufficient voltage can still transmit electrons. Loss remains irreversible
downstream, and the stop plane is sampled exactly even with coarse drawing
history. These are static states: switching edges, nanosecond pulse waveforms,
repetition frequency, RF operation and electron bunch formation are not
simulated. Do not infer temporal performance from this model.

The existing column solver propagates coordinates before applying interception
masks. Strongly deflected, already intercepted trajectories can therefore
produce numerical overflow warnings in downstream hexapole fields. Their
post-stop coordinates are excluded from transmitted rays, specimen currents
and images; early termination of their numerical propagation is not implemented.

## Geometry-aware condenser presets

Without the module, applying a mode uses the existing stored preset unchanged.
With the module installed, the stored strengths are an initial guess for the
actual assembled geometry. The mode application temporarily uses the open
state and 512 deterministic source rays, retaining the new physical stop.
Nanoprobe first refines the local C3 focus while retaining an acceptable
convergence, then the existing coupled C2/C3 solver validates the result.
Microprobe solves C2/C3 for illuminated diameter and near-parallel wavefront.
The user's blanking state and sampling choice are restored afterward.

| Mode | Requested quantity | Full-ray acceptance conditions |
| --- | --- | --- |
| Nanoprobe | 25 mrad current-weighted 95% semi-angle | Absolute log angle error ≤ 0.08 (approximately 8%); waist offset ≤ 10 nm |
| Microprobe | 2.0 µm current-weighted 95% diameter | Absolute log diameter error ≤ 0.05 (approximately 5%); absolute radial curvature ≤ 25 m⁻¹; 95% convergence ≤ 0.30 mrad and 99% convergence ≤ 0.50 mrad |

Both modes compare the observable at 0.10 mm and 0.05 mm integration steps,
requiring relative numerical spread ≤ 1%. The preliminary Nanoprobe C3 polish
aims at a tighter 0.001 nm waist residual; the full-ray acceptance conditions
remain those stated above. This additional focus refinement does not assert
picometre absolute physical accuracy.

Successful results are cached only against a fingerprint containing runtime
settings, absolute assembled positions and manifest contents. Geometry or
field-profile changes require a new solve. A failed recalculation restores
the previous mode, lens strengths, apertures and detector selections. Returned
mode metadata contains the newly measured quantities and drops old geometry's
probe and step-refinement metrics.

The principal corrector fields retain the selected mode's calibration. This
work recalculates condenser settings; it is not a new aberration-corrector
calibration or a manufacturer current table. C1 remains at its mode value;
the C1/C2 crossover catalog entry is not an additional enforced constraint in
this C2/C3 solve.

## Reproduction and measured default

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe scripts/calibrate_nanopulser_presets.py --rays 1000
```

The report is written to `outputs/nanopulser_calibration/report.json`. The
script leaves instrument and mode TOMLs unchanged. `--mode nano_probe` or
`--mode micro_probe` selects one mode; `--output` changes the report path.
The default validation used 300 kV, FEG, C3 + Probe Corrector, the 80 mm module
and a 100 µm C2 aperture. Calibration used 512 rays; the following independent
measurement used 1000 rays and the finer 0.025 mm integration step:

| Mode | C2 excitation | C3 excitation | Measured 95% semi-angle | Measured 95% diameter |
| --- | ---: | ---: | ---: | ---: |
| Nanoprobe | 25.25772717% | 21.27366657% | 24.119867 mrad | 0.453357 nm |
| Microprobe | 17.50823746% | 34.50480244% | 0.263760 mrad | 1.988809 µm |

Nanoprobe RMS radius was 0.106040 nm, waist offset 0.266291 nm and current
survival 53.8%. Halving the step from 0.05 to 0.025 mm changed its waist by
0.030412 nm. The threefold moment was 0.095916 in this finite ray sample;
these statistics do not establish a perfectly circular wave probe or a wave
FWHM. Microprobe radial wavefront curvature was −1.111873 m⁻¹ and survival
99.0%. Measured first applications took about 38 s for Nanoprobe and 12 s for
Microprobe on the development machine; timings depend on hardware and cache.

The numbers apply to that declared configuration. Altered sources, module
dimensions, apertures, voltage or other fields must pass their own live check;
the table is not a universal strength combination.
