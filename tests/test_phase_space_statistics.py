import numpy as np
import pytest
from temsim.physics.phase_space_statistics import weighted_phase_space_statistics


def test_covariance_is_weighted_population_not_ray_count_or_current():
    arrays = dict(x_m=np.array([-1., 1., 9.]), tx_rad=np.array([1., -1., 7.]),
                  y_m=np.array([2., -2., 8.]), ty_rad=np.array([.5, 1.5, 6.]),
                  weight=np.array([.1, .3, .6]), alive=np.array([True, True, False]))
    result = weighted_phase_space_statistics(arrays)
    assert result["transmitted_weight"] == pytest.approx(.4)
    assert result["effective_samples"] == pytest.approx(1.6)
    assert result["mean"][0] == pytest.approx(.5)
    assert result["covariance"][0, 0] == pytest.approx(.75)
    np.testing.assert_array_equal(arrays["weight"], [.1, .3, .6])
    assert result["phase_status"] == "NOT_COMPUTED"


def test_covariance_transforms_under_rotation_and_empty_is_unavailable():
    rng = np.random.default_rng(8)
    points = rng.normal(size=(100, 4))
    keys = ("x_m", "tx_rad", "y_m", "ty_rad")
    arrays = dict(zip(keys, points.T))
    arrays.update(weight=np.ones(100), alive=np.ones(100, bool))
    first = weighted_phase_space_statistics(arrays)
    angle = .37
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c,0,-s,0], [0,c,0,-s], [s,0,c,0], [0,s,0,c]])
    rotated = {**arrays, **dict(zip(keys, (points@rotation.T).T))}
    second = weighted_phase_space_statistics(rotated)
    np.testing.assert_allclose(second["covariance"], rotation@first["covariance"]@rotation.T, atol=1e-14)
    with pytest.raises(ValueError, match="No finite"):
        weighted_phase_space_statistics({**arrays, "alive": np.zeros(100, bool)})


def test_nonfinite_population_total_is_rejected():
    arrays = dict(x_m=[0., 1.], y_m=[0., 1.], tx_rad=[0., 1.], ty_rad=[0., 1.],
                  weight=[1.e308, 1.e308], alive=[True, True])
    with pytest.raises(ValueError, match="weight exceeds"):
        weighted_phase_space_statistics(arrays)


def test_moment_readout_is_scaled_read_only_and_cleared(qtbot, monkeypatch):
    from temsim.gui.sampling_panel import SamplingPanel
    panel = SamplingPanel()
    qtbot.addWidget(panel)
    monkeypatch.setattr("temsim.physics.simulation.run",
                        lambda *a, **k: pytest.fail("Displaying moments must not run transport"))
    cov = np.eye(4)*1e-18
    stats = dict(mean=[1e-9, 2e-3, -3e-9, -4e-3], covariance=cov,
                 rms_emittance_m_rad=[2e-12, 3e-12])
    panel._show_moments(stats)
    assert [panel.moments.item(0, i).text() for i in range(4)] == ["1", "2", "-3", "-4"]
    assert "X 2 | Y 3 nm mrad" in panel.emittance.text()
    assert panel.moments.item(1, 0).text() == "1"
    panel.set_checkpoint(None)
    assert panel.moments.item(0, 0) is None
    assert "unavailable" in panel.emittance.text()
