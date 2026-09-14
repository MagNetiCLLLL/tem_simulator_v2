"""Absorbed histories retain identity without invalidating live trajectories."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.optical_tuning import tuning_metrics


@pytest.mark.parametrize("invalid_row", [0, 1, 2])
def test_tuning_rejects_nonfinite_only_up_to_the_physical_stop(invalid_row):
    state = SimpleNamespace(projector_mode="diffraction",
        sample=SimpleNamespace(inserted=True, specimen_mode="atomic"))
    x = np.zeros((3, 1))
    x[invalid_row, 0] = np.nan
    branch = SimpleNamespace(z=np.arange(3.), x=x, y=np.zeros_like(x),
        tx=np.zeros_like(x), ty=np.zeros_like(x), blocked_z=np.array([1.]),
        alive=np.array([False]), ray_weight=np.ones(1))
    if invalid_row <= 1:
        with pytest.raises(ValueError, match="Non-finite tuning trajectory"):
            tuning_metrics(state, branch)
    else:
        assert tuning_metrics(state, branch)["sample_beam_surviving_rays"] == 0
