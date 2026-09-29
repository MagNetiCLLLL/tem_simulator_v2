# Explicit CIF test inputs

These files are regression-test inputs, not a runtime sample library. The
application starts in vacuum and accepts material only through a user import.

`Si.cif` is the unmodified user-supplied file from 2026-09-08 (a = 5.44370237 Å).
`Au.cif` and the original Si/Au TOML metadata are retained without modification
for provenance. The application does not read these metadata sidecars.

Tests specify their own orientation, insertion and numerical/material settings.
These fixtures must not be packaged as built-in application samples.
