import numpy as np
import pytest
from types import SimpleNamespace

import temsim.detector.eels_forward as eels_forward
from temsim.optics.column import default_state
from temsim.optics.energy_filter import (
    configure_energy_filter_operating_mode,
    ensure_energy_filter,
)
from temsim.optics.energy_filter_raytrace import simulate_energy_filter
from temsim.specimen.inelastic import real_inelastic_distribution


def test_raw_axis_mapping_keeps_zero_loss_peak_on_zero_starting_axis():
    mapped, omitted = eels_forward._map_raw_to_axis(
        np.asarray((1.0, 0.0, 0.0)),
        1.0,
        np.asarray((0.0, 1.0, 2.0, 3.0)),
    )

    assert mapped == pytest.approx((1.0, 0.0, 0.0))
    assert omitted == 0.0


def test_raw_axis_mapping_accounts_for_both_narrow_window_boundaries():
    raw = np.asarray((0.40, 0.30, 0.20, 0.10, 0.05))
    edges = np.asarray((1.0, 2.0, 3.0))

    mapped, omitted = eels_forward._map_raw_to_axis(raw, 1.0, edges)

    # 1 eV lies on the included lower edge; 3 eV lies on the included final
    # edge.  The 0 and 4 eV masses stay omitted instead of being clipped into
    # the boundary bins.
    assert mapped == pytest.approx((0.30, 0.30))
    assert omitted == pytest.approx(0.45)
    assert mapped.sum() + omitted == pytest.approx(raw.sum())
    raw_inside_first_moment = 0.30 * 1.0 + 0.20 * 2.0 + 0.10 * 3.0
    mapped_first_moment = float(np.dot(mapped, (1.5, 2.5)))
    assert abs(mapped_first_moment - raw_inside_first_moment) <= (
        0.5 * np.diff(edges)[0] * mapped.sum()
    )


def test_forward_axis_and_broadening_losses_are_explicitly_accounted():
    state = default_state()
    distribution = real_inelastic_distribution(state)
    edges = np.arange(0.0, 80.5, 0.5)
    centres = 0.5 * (edges[:-1] + edges[1:])

    result = eels_forward.simulate_eels_forward(
        state,
        distribution=distribution,
        energy_edges_ev=edges,
        spectrometer=eels_forward.ideal_spectrometer_transmission(centres),
        detector_response=eels_forward.EELSDetectorResponse(
            energy_resolution_fwhm_ev=2.0
        ),
    )

    metrics = result.metrics
    assert result.generated_components["zero_loss"].sum() > 0.0
    assert metrics["axis_mapping_omitted_probability"] >= 0.0
    assert metrics["raw_axis_omitted_probability"] >= 0.0
    assert metrics["energy_axis_omitted_probability"] == pytest.approx(
        metrics["raw_axis_omitted_probability"]
        + metrics["axis_mapping_omitted_probability"],
        abs=2.0e-12,
    )
    assert (
        result.generated_probability.sum()
        + metrics["energy_axis_omitted_probability"]
        + result.absorbed_probability
    ) == pytest.approx(1.0, abs=2.0e-12)
    assert abs(metrics["energy_axis_probability_accounting_residual"]) < 2.0e-15
    assert metrics["source_broadening_edge_loss_probability"] > 0.0
    assert metrics["detector_broadening_edge_loss_probability"] > 0.0


def test_forward_chain_uses_shared_distribution_and_conserves_components(
    monkeypatch,
):
    state = default_state()
    distribution = real_inelastic_distribution(state)
    edges = np.arange(-5.25, 500.26, 0.5)
    transfer = eels_forward.ideal_spectrometer_transmission(
        0.5 * (edges[:-1] + edges[1:])
    )
    monkeypatch.setattr(
        eels_forward,
        "real_inelastic_distribution",
        lambda _state: (_ for _ in ()).throw(AssertionError("recomputed")),
    )

    result = eels_forward.simulate_eels_forward(
        state,
        distribution=distribution,
        energy_edges_ev=edges,
        spectrometer=transfer,
    )

    component_sum = sum(
        result.generated_components.values(),
        np.zeros_like(result.generated_probability),
    )
    assert result.generated_probability == pytest.approx(component_sum)
    assert result.metrics["specimen_distribution_source"] == (
        "shared_specimen_interaction_result"
    )
    assert "zero_loss" in result.generated_components
    assert "plural_scattering" in result.generated_components
    assert result.generated_probability.sum() + result.absorbed_probability == (
        pytest.approx(1.0, abs=2.0e-6)
    )


def test_eels_and_eftem_use_their_own_active_detector_boundaries():
    state = default_state()
    ensure_energy_filter(state)
    state.energy_filter.enabled = True
    axis = np.linspace(-20.0, 20.0, 161)

    configure_energy_filter_operating_mode(state.energy_filter, "eftem")
    eftem = eels_forward.first_order_energy_filter_transmission(state, axis)
    assert not state.energy_filter.zebra_detector.inserted
    assert state.energy_filter.output_detector_inserted
    assert np.any(eftem.transmission > 0.0)
    assert "eftem_output_image_plane" in eftem.boundaries

    configure_energy_filter_operating_mode(state.energy_filter, "eels")
    eels = eels_forward.first_order_energy_filter_transmission(state, axis)
    assert state.energy_filter.zebra_detector.inserted
    assert not state.energy_filter.output_detector_inserted
    assert np.all(eels.transmission == 1.0)
    assert "zebra_active_state" in eels.boundaries


def test_eftem_image_uses_same_forward_spectrum_without_full_cube():
    state = default_state()
    distribution = real_inelastic_distribution(state)
    edges = np.arange(-5.25, 100.26, 0.5)
    centres = 0.5 * (edges[:-1] + edges[1:])
    transmission = eels_forward.SpectrometerTransmission(
        centres,
        ((centres >= 10.0) & (centres <= 20.0)).astype(float),
        "test_window",
        "test calibrated window",
        ("eftem_output_image_plane",),
    )
    forward = eels_forward.simulate_eels_forward(
        state,
        distribution=distribution,
        energy_edges_ev=edges,
        spectrometer=transmission,
    )
    image = np.asarray(((1.0, 2.0), (3.0, 4.0)))

    compact = eels_forward.eftem_image_from_shared_spectrum(image, forward)
    cube = eels_forward.eftem_spectral_cube_from_image(image, forward)

    assert compact == pytest.approx(
        eels_forward.apply_eftem_window(cube, transmission)
    )


def test_energy_filter_runtime_exposes_forward_result_from_same_distribution():
    state = default_state()
    ensure_energy_filter(state)
    state.energy_filter.enabled = True
    configure_energy_filter_operating_mode(state.energy_filter, "eels")
    distribution = real_inelastic_distribution(state)
    branch = SimpleNamespace(
        z=np.asarray((0.0, 1.0)),
        x=np.zeros((2, 1)),
        y=np.zeros((2, 1)),
        tx=np.zeros((2, 1)),
        ty=np.zeros((2, 1)),
        blocked_z=np.asarray((np.nan,)),
        energy_offset_ev=np.zeros(1),
        ray_weight=np.ones(1),
    )
    simulation = SimpleNamespace(
        incident=branch,
        branches={},
        metrics={},
        gun_trace=None,
    )

    result = simulate_energy_filter(state, simulation, distribution)

    assert result.eels_forward is not None
    assert result.eels_forward.metrics["specimen_distribution_source"] == (
        "shared_specimen_interaction_result"
    )
