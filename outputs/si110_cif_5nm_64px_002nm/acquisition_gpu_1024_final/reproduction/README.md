# Acquisition implementation snapshot

`implementation_snapshot.zip` includes all Python sources under `src/temsim`,
both acquisition scripts, the operating-mode catalog and `pyproject.toml`.
Every one of the 83 source hashes recorded at acquisition startup and the
generation-script hash was verified against the archive. The per-entry
digests and archive digest are in `archive_verification.json`.

`pip-freeze.txt` records the exact local Python environment. For the optional
GPU runtime, the installation command used with this project's `.venv` was:

```powershell
.venv\Scripts\python.exe -m pip install 'cupy-cuda12x[ctk]>=14,<15' 'cuda-toolkit==12.6.3'
```

The captured environment resolved CuPy 14.2.0, local CUDA toolkit 12.6.3 and
NVRTC 12.6.85. It used the existing NVIDIA 560.94 driver on a GTX 1060 3GB;
no system driver or system CUDA installation was changed. GPU package versions
are also recorded in `parameters.json`. The range command above describes the
installation procedure; use the recorded exact versions when recreating this
environment later.

Use an isolated project copy if restoring archived sources or instrument
TOMLs; extraction into a working project would overwrite its newer changes.
The parent acquisition directory contains its exact CIF, full format-5
operating profile, calibrated real-lens strengths and selected instrument
TOMLs. The command to rerun is in the acquisition index one directory above
that folder. Results remain a finite-grid, four-configuration calculation;
bitwise agreement across different numerical runtimes is not claimed.

## Profile export verification

The App-facing `operating_profile.toml` removes only two unsupported export
fields: `devices.simulation.last_gun_waist_mm` and
`devices.descan_deflector.wobble_enabled`. Its original bytes remain in
`operating_profile_raw.toml`. Loading the cleaned file reports no skipped
fields and produces exactly the same state as loading the raw file.

The normal `calibrate_scan_system` step, also invoked by the production
simulation, restores the derived dynamic Descan lower-coil gain. After that
step the sample, complete lenses, apertures, stigmators, deflectors, recording
planes, correctors, energy filter, scan controls, aberrations and numerical
settings match the acquisition inputs. The comparison and both profile hashes
are in `profile_reload_verification.json`; relative/absolute tolerances are
1e-12/1e-14. The actual source archive and recorded acquisition state were not
changed by this export-only cleanup.
