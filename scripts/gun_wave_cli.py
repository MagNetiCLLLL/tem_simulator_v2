"""Numerical controls for the executed surface-tip/gun chain, not a source."""
from temsim.physics.radial_gun_wave import RadialGunNumerics
from temsim.physics.radial_coordinates import RadialCoordinateBlend
from temsim.physics.wave_following_chart import WaveFollowingNumerics
from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement


def add_gun_wave_arguments(parser):
    group = parser.add_argument_group("Surface-tip gun discretisation (physical profile is unchanged)")
    defaults = RadialGunNumerics()
    group.add_argument("--surface-element-order", type=int, choices=(1, 2), default=1)
    group.add_argument("--radial-modes", type=int, default=defaults.radial_modes)
    group.add_argument("--radial-quadrature", type=int, default=defaults.potential_quadrature)
    group.add_argument("--relative-axial-step", type=float, default=defaults.relative_axial_step)
    group.add_argument("--axial-integrator", choices=("midpoint", "cf4", "adaptive_cf4"), default=defaults.axial_integrator)
    group.add_argument("--coordinate-width", type=float, default=defaults.coordinate_width_over_cap_radius,
                       help="Numerical basis width / emitting-cap radius; does not change the physical tip")
    group.add_argument("--coordinate-blend", type=float, nargs=3, metavar=("WIDTH", "START_MM", "END_MM"))
    group.add_argument("--wave-following-iterations", type=int, default=0)
    group.add_argument("--wave-following-phase-order", type=int, choices=(2, 4), default=2)
    group.add_argument("--chart-smoothing-log-z", type=float, default=0.)
    group.add_argument("--occupied-axial-tolerance", type=float,
                       help="Enable source-specific two-way refinement at this total complex-amplitude tolerance")
    group.add_argument("--occupied-axial-evaluations", type=int, default=200_000)
    group.add_argument("--occupied-refinement-strategy", choices=("local_sum", "global_embedded", "spatial_embedded"), default="local_sum")
    group.add_argument("--occupied-axial-rounds", type=int, default=12)
    group.add_argument("--occupied-mesh-seed", help="Numerical subdivisions only; not a source or cached wave")
    group.add_argument("--occupied-mesh-energy-ev", type=float,
                       help="Exact emission energy for these numerical hints; all other energies still execute")
    group.add_argument("--occupied-axial-workers", type=int, default=1)
    group.add_argument("--occupied-axial-executor", choices=("thread", "process"), default="thread")
    group.add_argument("--occupied-axial-integrator", choices=("cf4", "cf6"), default="cf4")
    group.add_argument("--occupied-budget-allocation", choices=("uniform", "initial_indicator"), default="uniform")


def gun_wave_numerics(args, gun):
    """Bind the same complete numerics into execution and checkpoint identity."""
    from temsim.physics.axial_mesh_seed import read_mesh_seed
    return RadialGunNumerics(radial_modes=args.radial_modes, potential_quadrature=args.radial_quadrature,
        relative_axial_step=args.relative_axial_step, axial_integrator=args.axial_integrator,
        field_step_mm=gun.field_step_mm, bore_step_mm=gun.bore_step_mm,
        maximum_steps=gun.max_steps, maximum_working_bytes=gun.maximum_checkpoint_bytes,
        coordinate_width_over_cap_radius=args.coordinate_width,
        coordinate_blend=None if args.coordinate_blend is None else RadialCoordinateBlend(*args.coordinate_blend),
        wave_following=WaveFollowingNumerics(iterations=args.wave_following_iterations,
            phase_order=args.wave_following_phase_order, smoothing_log_z_width=args.chart_smoothing_log_z),
        occupied_refinement=OccupiedAxialRefinement(enabled=args.occupied_axial_tolerance is not None,
            initial_mesh=read_mesh_seed(args.occupied_mesh_seed),
            initial_mesh_energy_ev=args.occupied_mesh_energy_ev,
            strategy=args.occupied_refinement_strategy, maximum_rounds=args.occupied_axial_rounds,
            tolerance=.001 if args.occupied_axial_tolerance is None else args.occupied_axial_tolerance,
            maximum_evaluations=args.occupied_axial_evaluations, workers=args.occupied_axial_workers,
            executor=args.occupied_axial_executor, integrator=args.occupied_axial_integrator,
            error_budget_allocation=args.occupied_budget_allocation,
            maximum_working_bytes=gun.maximum_checkpoint_bytes)).validate()
