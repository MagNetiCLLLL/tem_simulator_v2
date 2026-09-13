"""Independent complex-field convergence of the numerical moving frame."""
import importlib.util
from pathlib import Path


def test_moving_frame_converges_to_nonparaxial_bessel_angular_spectrum():
    from threadpoolctl import threadpool_limits
    path = Path(__file__).parents[1]/"scripts"/"inspect_moving_radial_frame.py"
    spec = importlib.util.spec_from_file_location("moving_frame_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with threadpool_limits(1):
        errors = [module.calculate(step)["relative_complex_error"] for step in (10., 5., 2.5)]
    assert errors[-1] < 1.4e-5
    assert all(coarse/fine > 3.8 for coarse, fine in zip(errors, errors[1:]))
