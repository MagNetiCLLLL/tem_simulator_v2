# Reproduction snapshot

`implementation_snapshot.zip` contains the actual Python sources under
`src/temsim` and `scripts`, the operating-mode catalog and `pyproject.toml`.
Every one of the 83 numeric/source hashes recorded at the first constituent's
startup, plus the acquisition script hash, matched the archive. The archive
and per-entry hashes are recorded in `archive_verification.json`.

The experiment uses four independent single-phonon acquisitions in sequence,
with seeds 707, 708, 709 and 710. Each `seed_*` directory retains its exact
profile, original profile, archived CIF, selected instrument TOMLs, full state,
source hashes, progress and raw arrays. The ensemble is the arithmetic mean of
the intensities. It is statistically comparable to a four-configuration run,
but it does not use the same random-realisation sequence as one seed with four
configurations. Use the runner to reproduce this recipe:

```powershell
.venv\Scripts\python.exe scripts/run_stem_profile_ensemble.py --profile outputs/haadf_dpa_clearance/si110_haadf_dpa_12mm.toml --output outputs/si110_underfocus_100nm/reproduction_run --seeds 707 708 709 710 --scan-side 16 --step-nm .08 --grid 2048 --fov-angstrom 80 --padding-factor 1.3 --effective-c1-nm -100
```

The ray probe is measured separately for each constituent. Configured C1 is
chosen so that ray C1 plus configured C1 equals -100 nm: the coherent probe's
focus lies 100 nm downstream of the specimen. No lens excitation or incident
ray coordinates are overwritten. The original profile selects the actual
12 mm DPA-to-HAADF geometry and calibrated real post-sample lenses.

`pip-freeze.txt` records the local Python packages. The optional GPU runtime was
installed locally using:

```powershell
.venv\Scripts\python.exe -m pip install 'cupy-cuda12x[ctk]>=14,<15' 'cuda-toolkit==12.6.3'
```

This environment resolved CuPy 14.2.0, CUDA toolkit 12.6.3 and NVRTC 12.6.85,
using the existing NVIDIA 560.94 driver. No system driver or system CUDA
installation was changed. Use the recorded exact package versions when
recreating the environment. Restoring the archive and instrument TOMLs should
be done in an isolated project copy, since extraction can overwrite newer
local project changes. Bitwise agreement across numerical runtimes is not
claimed.
