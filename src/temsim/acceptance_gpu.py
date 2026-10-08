"""Reviewed real-device checks that cannot establish GPU evidence on CPU CI.

These cases stay collected and their failures still fail the lane. Only a
normal pytest skip may be recorded separately when the runner explicitly
allows GPU skips. Policy/emulation tests are deliberately not included.
"""

GPU_TEST_FUNCTIONS = {
    "test_compute_backend.py": (
        "test_auto_cuda_ray_trace_matches_cpu_with_energy_spread",
    ),
    "test_canonical_action.py": (
        "test_cuda_lct_and_gauge_keep_complete_complex_field_on_device",
    ),
    "test_electrostatic_column_transport.py": (
        "test_cuda_electric_batches_preserve_every_ray_and_report_real_completed_counts",
    ),
    "test_galerkin_potential.py": (
        "test_gpu_galerkin_matches_dense_complex_action_without_iteration_transfers",
        "test_gpu_galerkin_resource_and_physics_failures_do_not_fallback",
        "test_gpu_execution_retry_uses_original_host_wave_only_for_resource_failure",
    ),
    "test_galerkin_specimen.py": (
        "test_gpu_material_phase_interpolation_and_bandlimit_preserve_carrier_and_weights",
        "test_gpu_material_restarts_full_quadrature_and_phase_after_resource_failure",
        "test_gpu_repeated_material_operators_release_private_split_blocks",
        "test_gpu_retired_material_pool_preserves_live_device_output",
    ),
    "test_gpu_capture_contract.py": (
        "test_actual_gpu_capture_matches_cpu_cube_and_detector_reintegration",
    ),
    "test_multislice.py": (
        "test_cupy_multislice_matches_complex128_cpu_reference",
    ),
    "test_stem_cuda_pipeline.py": (
        "test_explicit_cuda_keeps_stem_arrays_resident_until_one_bulk_transfer",
        "test_resident_cuda_frozen_phonon_detector_signals_match_cpu_reference",
        "test_resident_cuda_projected_phase_object_matches_cpu_reference",
        "test_dynamic_detector_centres_match_cpu_even_after_partial_cuda_resource_failure",
    ),
    "test_wave_device.py": (
        "test_real_gpu_column_matches_cpu_complex_field_and_host_checkpoint",
        "test_real_gpu_physical_annulus_and_column_aperture_masks",
    ),
    "test_wave_fft.py": (
        "test_cupy_tem_image_fft_matches_numpy_reference",
        "test_cupy_stem_detector_fft_matches_numpy_reference",
    ),
    "test_wave_grid.py": (
        "test_cuda_fourier_refinement_and_nonlinear_carriers_preserve_phase_and_rejections",
    ),
    "test_tilted_column_wave.py": (
        "test_real_gpu_tilted_objective_matches_cpu_complex_checkpoint",
        "test_real_column_higher_order_magnetic_residual_runs_on_cpu_and_gpu",
    ),
    "test_tilted_spherical_column.py": (
        "test_real_gpu_tilted_cs_matches_cpu_and_mechanical_ray_response",
        "test_captured_default_electric_tail_tilted_cs_cpu_gpu",
    ),
    "test_posed_aberration_wave.py": (
        "test_real_gpu_tilted_screen_matches_cpu_absolute_complex_field",
    ),
    "test_wave_magnetic_residual.py": (
        "test_actual_gpu_matches_cpu_full_complex_envelope",
    ),
    "test_posed_multipole_wave.py": (
        "test_real_gpu_native_multipole_residual_matches_cpu",
    ),
    "test_tilted_multipole_column.py": (
        "test_actual_tilted_stigmator_gpu_and_cpu_wave_match_particle_centroid",
    ),
    "test_posed_wave_aperture.py": (
        "test_gpu_uses_same_swept_geometry_and_contacts",
    ),
    "test_posed_wave_hardware.py": (
        "test_real_cuda_bore_mask_matches_ray_contacts_and_cpu",
    ),
    "test_tilted_aperture_column.py": (
        "test_closed_tilted_plate_only_absorbs_local_slab_on_actual_gpu",
        "test_tilted_c2_finite_shoulder_diffraction_cpu_gpu_complex_parity",
        "test_tilted_c2_broad_beam_total_transmission_stabilizes_on_gpu",
    ),
}
GPU_TEST_IDS = frozenset(
    f"tests/{filename}::{name}"
    for filename, names in GPU_TEST_FUNCTIONS.items()
    for name in names
)
# This test also has mandatory CPU parameters; never exclude its whole function.
GPU_PARAMETER_IDS = frozenset({
    "tests/test_electrostatic_column_transport.py::"
    "test_closed_electric_and_magnetic_actions_match_reference_backend[CUDA GPU]",
})


def is_gpu_hardware_test(node: str) -> bool:
    return node.split("[", 1)[0] in GPU_TEST_IDS or node in GPU_PARAMETER_IDS


def is_complete_skip(outcomes: dict) -> bool:
    """Neither missing phases nor failed teardown are an optional-device skip."""
    return outcomes in (
        {"setup": "skipped", "teardown": "passed"},
        {"setup": "passed", "call": "skipped", "teardown": "passed"},
    )


def is_gpu_unavailable_reason(reason: str) -> bool:
    """Match reviewed device/dependency admission messages, not arbitrary skips."""
    reason = reason.removeprefix("Skipped: ")
    return reason in {
        "CUDA device unavailable", "NOT_RUN: actual CUDA device unavailable",
        "No CUDA hardware", "CuPy CUDA backend unavailable", "No CUDA device",
        "CUDA is unavailable", "A CUDA device is required for complex128 parity",
        "optional real CUDA verification requires CuPy", "CuPy found no CUDA device",
    } or reason.startswith((
        "could not import 'cupy':", "CUDA runtime unavailable:", "CuPy CUDA unavailable:",
    ))
