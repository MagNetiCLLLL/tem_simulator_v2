# Si[110] STEM with DF direct-disk clearance

Fresh simultaneous HAADF/DF/BF CUDA multislice acquisition: 32 × 32 pixels, 0.04 nm pitch, 1.28 nm pixel FOV. The archived 300 kV illumination, focus, lens excitations and detector Z positions are retained. Only DF ID/OD are changed to 60/100 mm in an independent validated instrument catalog. The original catalog listed modules absent from its archive; the derived catalog lists only the three archived installed modules. No global TOML was edited.

Si is the exact archived user CIF, diameter 10 nm, thickness 5 nm, [110], explicit 0.085 Å one-axis frozen-phonon RMS; 4 configurations, seed 707. Wave grid 1024², 4 nm FOV. Current production numerical code is archived here, including the corrected sample-boundary Rutherford overlap; this is a new calculation, not reuse of the historical integrated DF image. The coherent wave plus approximate screened-Rutherford high-angle tail and complete sequential physical stops are retained. No shadow correction is applied.

mask_preflight.json records the physical Jdiff matrices and sampled acceptance at all raster positions; angles ≤27 mrad have no DF hits. Reference collection angles are specific to this archived camera-length calibration, not universal detector specifications. This 32-pixel raster is a preview at coarser spatial sampling than the historical 64-pixel image; it does not establish wave-grid or phonon convergence.

raw_scan.npz contains source-normalized fractions, coherent/tail parts, loss budget and separate expected-electron/Poisson-count arrays. The 100 pA, 10 µs per-pixel, seed 42 counting exposure is readout-only and does not change the acquisition period or traced illumination. It assumes ideal independent electron detection and omits efficiency, electronic noise and saturation. NPZ/TIFF row zero is minimum Y; PNG row zero is maximum Y. Fraction PNGs use individual min/max; counts PNGs use zero-to-labelled-maximum. No smoothing or sharpening.

Reproduce with the archived source/scripts, local .venv dependencies in pip_freeze.txt, and this script's --archive path pointing at the original self-contained acquisition archive. The exported profile must be loaded with this directory's instrument_inputs catalog to retain DF geometry. Loading the profile alone in the ordinary GUI does **not** install these dimensions: the GUI still owns its global catalog. Opening the copied TOML in the 3D editor alone is also not an assembly installation.

## Completed acquisition and verification

The simultaneous three-channel acquisition took **430.7554 s (7 min 10.8 s)**. Actual execution used the resident CuPy CUDA pipeline, 1024 × 1024 wave pixels, 40 Å window, 4 frozen-phonon configurations (seed 707). The archived incidence was re-traced: 561/1000 surviving rays, 24.80467236 mrad 95% semiangle, 26.37509739 mrad angular edge, and effective C1 = +0.0836587533 nm. No focus correction was applied.

DF is at the unchanged z = 2804.15 mm. Its actual Jdiff matrix in m/rad is [[−0.9910601974, −0.1245080680], [0.1245080680, −0.9910601974]], yielding **30.03452245–50.05753742 mrad** reference acceptance. HAADF remains 60.00–330.00 mrad; BF remains 0–0.56788512 mrad. The 1024-position, 360-azimuth polar mask check found zero DF hits at radii 0–27 mrad and complete acceptance within the tested 32–48 mrad interior, including all actual upstream stops. This is a finite sampling check; reference matrices and detailed stops are saved in mask_preflight.json.

| Channel | Mean source fraction | Mean expected e−/pixel at 100 pA, 10 µs | Mean sampled e−/pixel | Tail share |
|---|---:|---:|---:|---:|
| HAADF | 0.00080167245 | 5.00365 | 4.94629 | 9.36765% |
| DF | 0.00266647700 | 16.64284 | 16.83105 | 0% |
| BF | 0.00035545528 | 2.21858 | 2.21582 | 0% |

The counting exposure represents 6241.5091 emitted source electrons per pixel, not 6241 electrons reaching the sample. The original 0.561 sample-transmission factor is already included in the saved fractions. Acquisition raster period remains **1 s**; the independent counting exposure corresponds to 10.24 ms for 1024 pixels but was not substituted into the time-dependent optical calculation. These low expected counts make a single Poisson realization visibly noisy.

acquisition_verification.json records passing checks for all 20 NPZ arrays, fraction and count TIFFs, vertical PNG orientation and transfer functions, exact seed-42 Poisson regeneration, finite/nonnegative/nonconstant images, coherent+tail reconstruction, and probability conservation (maximum error 2.22 × 10⁻¹⁶). It also verifies the exact user CIF hash, every numerical source hash in the source archive, unchanged non-DF module fields, and zero-skipped-fields profile import with the independent catalog. The restored lenses, specimen, wave settings, DF dimensions and reference angles match. The verification script runs no new ray or wave simulation.

From the project root, the original acquisition command is:

```powershell
.venv\Scripts\python.exe scripts/run_df_clearance_scan.py --output outputs/df_direct_beam_clearance/a_fresh_reproduction_directory
```

The script's default archive is recorded in parameters.json. A relocated input archive can be supplied with `--archive PATH`. The actual script and numerical source bytes used for this acquisition are preserved in implementation_snapshot.zip; pip_freeze.txt records its runtime dependencies. The new final output directory itself also contains the required parameters.json, operating_profile.toml, input_Si.cif and independent instrument_inputs for specifying it as an explicit archive of this physical setup.
