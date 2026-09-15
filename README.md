# TEM Simulator v2

TEM Simulator v2 is a Python desktop application for exploring transmission
electron microscopy and electron optics. It provides an interactive environment
for studying how an electron source, microscope geometry and operating settings
influence beam propagation and recorded signals.

The project is intended for teaching, experimentation and model development.
Instrument definitions are stored in editable TOML files, with a PySide6 interface
for configuring and visualising the microscope. Models include explicit
approximations and are not a calibrated replica of a commercial instrument.

The current development focus is classical particle transport. Coherent
tip-to-column wave development is paused; existing wave models and historical
results remain available within their documented limits.

## Getting started

Use 64-bit Python 3.12 on Windows:

```powershell
python setup_env.py
.venv\Scripts\python.exe main.py
```

## Documentation

See [docs/](docs/) for workflows, model assumptions and development notes, and
[configs/](configs/) for instrument definitions and operating settings.

## License

[MIT License](LICENSE).
