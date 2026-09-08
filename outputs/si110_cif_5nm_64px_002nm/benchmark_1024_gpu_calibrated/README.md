# Si [110] STEM acquisition

This is a 4×4 performance sample, not the requested final 64×64 image.

The specimen is the archived user CIF, oriented [110] along the beam and [1 -1 0] along X. The actual production nanoprobe/diffraction preset is used with a 0.05 mm ray-integration step. No incident coordinates or slopes were edited.

Open `operating_profile.toml` in the App to restore inputs. `raw_scan.npz` retains unnormalised source-probability fractions, coherent and tail contributions, and actual raster coordinates. TIFFs contain float32 fractions. The 16-bit PNGs map each channel's min/max to 0/65535; exact mapping is in metrics.json. The comparison figure uses independently labelled probability scales.

Finite-grid elastic frozen-phonon multislice and the structure-derived screened Rutherford high-angle extension are separate models. The extension is not a Mott calculation or recovered high-angle multislice interference. The displayed cutoff and detector tail fractions disclose that distinction. Four configurations are a finite ensemble, not an established convergence study. The explicit Si RMS assumption is 0.085 Å and is not a thermal parameter measured from this CIF. No shot noise is added.

Parameters, script/source hashes, physical detector angles and hardware are in parameters.json; production diagnostics and image statistics are in metrics.json. The archived instrument_inputs preserve the selected input TOMLs.
