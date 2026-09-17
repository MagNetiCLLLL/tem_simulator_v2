"""Detached, bounded illumination calibration for assembled particle columns.

First-order proposals are seeds only. Accepted settings require full physical
tip/column transport at the upper specimen surface. No downstream source,
geometry alteration, coherent gun solve or image computation is introduced.
"""
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from temsim import input_io
import tomllib

import numpy as np
from scipy.optimize import least_squares, brentq
from threadpoolctl import threadpool_limits

from temsim.assembly_catalog import AssemblySelection
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.operating_modes import mode_by_key, _apply_values
from temsim.optics.surface_probe_focus import measure_surface_focus, surface_z_mm
from temsim.optics.illumination_current import aperture_gate, current_gate, set_calibration_flux


@dataclass(frozen=True)
class IlluminationTarget:
    key: str
    alpha95_mrad: float
    diameter95_um: float | None
    relative_tolerance: float = .01
    focus_tolerance_nm: float = 1.
    maximum_curvature_per_m: float = 25.

    def accepts(self, measurement):
        s = measurement.statistics
        if self.key == "nano_probe":
            return measurement.accepts(self.alpha95_mrad,
                angle_relative_tolerance=self.relative_tolerance,
                focus_tolerance_nm=self.focus_tolerance_nm)
        return bool(s.surviving_rays >= 16 and measurement.effective_rays >= 16
            and s.surviving_fraction > 0
            and 0 < s.convergence_95_mrad <= self.alpha95_mrad
            and abs(s.illumination_diameter_95_um / self.diameter95_um - 1) <= self.relative_tolerance
            and abs(s.radial_wavefront_curvature_per_m) <= self.maximum_curvature_per_m)

    def residual(self, statistics):
        s = statistics
        if s.surviving_rays < 16:
            raise ValueError("Fewer than 16 transmitted calibration rays; increase the sampling budget")
        if self.key == "nano_probe":
            values = (math.log(s.convergence_95_mrad / self.alpha95_mrad),
                      s.waist_offset_m / 1e-6)
        else:
            values = (math.log(s.illumination_diameter_95_um / self.diameter95_um),
                      s.radial_wavefront_curvature_per_m / 1e4)
        if not np.all(np.isfinite(values)):
            raise ValueError("Nonfinite illumination residual")
        return np.asarray(values)


def load_targets(path=None):
    from temsim.paths import OPERATING_MODE_CONFIG_ROOT
    path = Path(path) if path is not None else OPERATING_MODE_CONFIG_ROOT / 'illumination_targets.toml'
    with input_io.open_input(path) as stream:
        document = tomllib.load(stream)
    if document.get('schema') != 'assembly-illumination-targets-v1':
        raise ValueError('Unsupported illumination target schema')
    result = {}
    for key in ('nano_probe', 'micro_probe'):
        values = dict(document[key])
        values.setdefault('diameter95_um', None)
        target = IlluminationTarget(key, **values)
        positive = (target.alpha95_mrad, target.relative_tolerance,
                    target.focus_tolerance_nm, target.maximum_curvature_per_m)
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError('Illumination target values must be finite and positive')
        if key == 'micro_probe' and (target.diameter95_um is None
                or not math.isfinite(target.diameter95_um) or target.diameter95_um <= 0):
            raise ValueError('Microprobe requires a positive current-contained diameter')
        result[key] = target
    return result


TARGETS = load_targets()


def diameter_gate(state, measurement, mode_key, path=None):
    """A small geometric diameter is required, not a wave-resolution claim."""
    from temsim.paths import OPERATING_MODE_CONFIG_ROOT
    path = Path(path) if path is not None else OPERATING_MODE_CONFIG_ROOT / 'illumination_targets.toml'
    with input_io.open_input(path) as stream:
        limits = tomllib.load(stream)['probe_diameter']
    installed = {c.key for c in state._resolved_optics_layout}
    corrected = 'probe_hp1_hexapole' in installed and 'probe_hp2_hexapole' in installed
    limit = float(limits['corrected_maximum_nm' if corrected else 'uncorrected_maximum_nm'])
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('Probe-diameter limits must be finite and positive')
    diameter = float(measurement.statistics.illumination_diameter_95_um)*1e3
    passed = math.isfinite(diameter) and 0 <= diameter and (diameter < limit if corrected else diameter <= limit)
    return dict(required=mode_key == 'nano_probe', probe_corrector_installed=corrected,
                diameter95_nm=diameter, maximum_nm=limit, strict=corrected,
                passed=passed if mode_key == 'nano_probe' else True)


def assembly_combinations(catalog):
    """List every selectable assembly; do not infer downstream equivalence."""
    return tuple(AssemblySelection(g.name, c.name, r.name, b.name)
        for g in catalog.guns for b in catalog.beam_blankers
        for c in catalog.columns for r in catalog.recording_systems)


def seed_illumination(state, mode_key):
    """Apply historical operating values ONLY as an explicitly unqualified seed.

    Disabled/missing hardware must never be re-enabled by an old C3 preset.
    Source, gun voltages, downstream lenses and geometry remain unchanged.
    """
    from temsim.runtime_parameters import runtime_targets
    mode = mode_by_key(mode_key)
    targets = runtime_targets(state)
    end = surface_z_mm(state)
    installed = {c.key for c in state._resolved_optics_layout
                 if c.optical_reference_plane_z_mm is not None
                 and c.optical_reference_plane_z_mm <= state.sample.z_mm}
    devices = {}
    for key, values in mode.devices.items():
        if key not in installed or key not in targets:
            continue
        obj = targets[key].obj
        if not bool(getattr(obj, "enabled", True)):
            continue
        if getattr(obj, "z_mm", end) > state.sample.z_mm:
            continue
        devices[key] = dict(values)
    apertures = {k: dict(v) for k, v in mode.apertures.items()
                 if k in installed and k in targets}
    if mode_key == "nano_probe" and "condenser_aperture_2" in apertures:
        # Historical 100 um / 25 mrad pupil is a seed, not a 30 mrad
        # calibration. The physical 120 um opening is traced and clipped.
        apertures["condenser_aperture_2"]["diameter_mm"] = .12
    _apply_values(state, replace(mode, devices=devices, apertures=apertures))
    state.sync_objective()


def variable_lenses(state, mode_key):
    enabled = {l.key for l in state.lenses if l.enabled}
    if "mini_condenser" in enabled:
        # Keep the established condenser/corrector relay and its crossovers.
        # Both overlapping specimen-front fields remain physically executed.
        keys = ("mini_condenser", "objective_lens")
    elif "condenser_lens_3" in enabled:
        keys = ("condenser_lens_2", "condenser_lens_3")
    else:
        keys = (("condenser_lens_2", "objective_lens") if mode_key == "nano_probe"
                else ("condenser_lens_1", "condenser_lens_2"))
    if not set(keys) <= enabled:
        raise ValueError("The assembled column lacks the required focusing lenses")
    return keys


def _focus_branch(measure, target, initial, upper):
    """Solve focus first, then illumination along that same local branch.

    A joint two-dimensional fit wastes steps trading nanometre focus error
    against the angular target. The inner solve uses the real objective field,
    not an imposed focal-plane constraint or a rescaling of the ray state.
    """
    focused = {}
    objective = float(initial[1])

    def at(primary):
        nonlocal objective
        primary = float(primary)
        if primary in focused:
            return focused[primary]
        if focused:
            nearest = min(focused, key=lambda v: abs(v-primary))
            objective = focused[nearest][0][1]
        for _ in range(10):
            vector = np.array([primary, objective])
            m = measure(vector)
            error = (m.statistics.waist_offset_m if target.key == "nano_probe"
                     else m.statistics.radial_wavefront_curvature_per_m)
            tolerance = .05e-9 if target.key == "nano_probe" else .01
            if math.isfinite(error) and abs(error) <= tolerance:
                focused[primary] = (vector, m)
                return vector, m
            delta = 1e-4 if objective+1e-4 < upper[1] else -1e-4
            shifted = measure([primary, objective+delta]).statistics
            shifted_error = shifted.waist_offset_m if target.key == "nano_probe" else shifted.radial_wavefront_curvature_per_m
            derivative = (shifted_error-error)/delta
            if not math.isfinite(derivative) or abs(derivative) < 1e-20:
                raise ValueError("The objective focus derivative is unresolved")
            proposal = np.clip(objective - np.clip(error/derivative, -2., 2.), .001, upper[1]-.001)
            if abs(proposal-objective) < 1e-12:
                raise ValueError("The objective focus solve reached its configured limit")
            objective = float(proposal)
        raise ValueError("The local objective focus branch did not converge")

    def residual(primary):
        _, m = at(primary)
        value = m.statistics.convergence_95_mrad if target.key == "nano_probe" else m.statistics.illumination_diameter_95_um
        requested = target.alpha95_mrad if target.key == "nano_probe" else target.diameter95_um
        if m.statistics.surviving_rays < 16 or not math.isfinite(value) or value <= 0:
            raise ValueError('The focused branch has insufficient transmitted rays or unresolved illumination')
        return math.log(value/requested)

    observations = {float(initial[0]): residual(initial[0])}
    # Fit near the centre of the acceptance band, not its edge. Independent
    # particle samples must still pass the original (unchanged) target gate.
    if abs(observations[float(initial[0])]) <= 1e-4 and target.accepts(at(initial[0])[1]):
        return at(initial[0])[0]
    for distance in (.5, 1.5, 4., 8., 16., 32.):
        for sign in (-1., 1.):
            p = float(np.clip(initial[0]+sign*distance, .001, upper[0]-.001))
            if p in observations:
                continue
            observations[p] = residual(p)
            if abs(observations[p]) <= 1e-4 and target.accepts(at(p)[1]):
                return at(p)[0]
            ordered = sorted(observations)
            for a, b in zip(ordered[:-1], ordered[1:]):
                if observations[a]*observations[b] < 0:
                    root = brentq(residual, a, b, xtol=1e-7, maxiter=20)
                    return at(root)[0]
    raise ValueError("No illumination target bracket found on the searched focus branch")


def _proposals(state, keys, target, initial, maximum_evaluations, *, neighbouring_first=True):
    from temsim.optics.direct_alignment import _LiveFirstOrderModel
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from temsim.physics.beam_statistics import transverse_beam_statistics
    e = trace_source_to_exit(state).exit_bundle
    source = np.asarray([e.x_m, e.y_m, e.tx_rad, e.ty_rad])
    start, end = state.electron_gun.exit_plane_z_mm, surface_z_mm(state)
    apertures = [a for a in state.apertures if a.enabled and getattr(a, "installed", True)
                 and start < a.z_mm < end]
    if state.nanopulser.installed:
        apertures.append(state.nanopulser.aperture)
    apertures.sort(key=lambda a: a.z_mm)
    planes = [a.z_mm for a in apertures] + [end]
    model = _LiveFirstOrderModel(state, start, end, keys, step_mm=.1, capture_z_mm=planes)

    def residual(values):
        positions = model.matrices_at(values, planes) @ source
        alive = e.alive.copy()
        for aperture, rays in zip(apertures, positions[:-1]):
            if hasattr(aperture, "transmission_mask"):
                alive &= aperture.transmission_mask(rays[0]*1e3, rays[1]*1e3)
            else:
                alive &= np.hypot(rays[0]*1e3-aperture.offset_x_mm,
                                  rays[1]*1e3-aperture.offset_y_mm) <= aperture.radius_mm
        return target.residual(transverse_beam_statistics(*positions[-1], alive=alive, weights=e.weight))

    seeds = [initial]
    # Alternative branches are proposals, never silently selected as qualified.
    if target.key == "nano_probe":
        seeds += [np.array([v, initial[1]]) for v in (12., 25., 40., 55.)]
    else:
        seeds += [np.array([a, b]) for a, b in ((15., 30.), (30., 40.), (50., 20.))]
    found = [(math.inf, np.asarray(initial))]
    for seed in seeds:
        try:
            fit = least_squares(residual, np.clip(seed, .001, model.upper-.001),
                bounds=(np.zeros(2), model.upper), max_nfev=maximum_evaluations,
                x_scale="jac", diff_step=1e-6, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            if np.all(np.isfinite(fit.fun)):
                found.append((float(np.linalg.norm(fit.fun)), fit.x))
        except (ValueError, FloatingPointError):
            continue
    found.sort(key=lambda item: item[0])
    # Try the neighbouring branch first. A distant first-order solution can
    # have an extra crossover despite a smaller terminal residual.
    selected = [v for _, v in found[:2]]
    return ([np.asarray(initial)] + selected if neighbouring_first
            else selected + [np.asarray(initial)])


@input_io.using_state_inputs
def calibrate_illumination(state, mode_key, *, rays=193, maximum_evaluations=30,
                           controls=None, search_branches=False, c2_aperture_mm=None,
                           mini_polarity=None,
                           progress=lambda message: None):
    """Return a non-installed candidate and audit. Never qualify from a fit alone.

    This stage checks focus/angle and column-step convergence at a fixed ray
    budget. Gun-step, sampling, spot size and historical crossover topology
    remain separate publication gates and are stated explicitly in the audit.
    """
    targets = load_targets() if input_io.active_archive() is not None else TARGETS
    if mode_key not in targets or rays < 32 or maximum_evaluations < 1:
        raise ValueError("Select a supported mode and positive calibration budget (at least 32 rays)")
    target = targets[mode_key]
    original = capture_instrument_snapshot(state)
    candidate = original.restore()
    if candidate.electron_gun.source_representation != "classical_particles" or candidate.vacuum_map.enabled:
        raise ValueError("Default calibration requires classical tip emission with vacuum participation off")
    candidate.electron_gun.emitter.ray_count = int(rays)
    seed_illumination(candidate, mode_key)
    if mini_polarity is not None:
        if mini_polarity not in (-1, 1):
            raise ValueError('Mini condenser polarity must be -1 or +1')
        mini = next((l for l in candidate.lenses if l.key == 'mini_condenser' and l.enabled), None)
        if mini is None:
            raise ValueError('No installed mini condenser to set polarity')
        mini.polarity = int(mini_polarity)
    if c2_aperture_mm is not None:
        if not math.isfinite(c2_aperture_mm) or c2_aperture_mm <= 0:
            raise ValueError('The physical C2 aperture diameter must be finite and positive')
        candidate.condenser_aperture_2.diameter_mm = float(c2_aperture_mm)
    if not aperture_gate(candidate)['passed']:
        raise ValueError('The physical C2 aperture must be enabled with diameter 20-250 um')
    keys = tuple(controls) if controls is not None else variable_lenses(candidate, mode_key)
    lenses = {l.key: l for l in candidate.lenses}
    if len(keys) != 2 or len(set(keys)) != 2 or any(k not in lenses or not lenses[k].enabled
            or lenses[k].z_mm > candidate.sample.z_mm for k in keys):
        raise ValueError("Select two distinct installed pre-specimen focusing lenses")
    initial = np.asarray([lenses[k].percent for k in keys])
    upper = np.asarray([lenses[k].max_percent for k in keys])
    report = dict(status="NOT_QUALIFIED", targets=asdict(target),
        input_digest=original.physical_digest, rays=rays, variable_lenses=keys,
        pending_gates=["gun_step", "particle_sampling", "probe_diameter", "crossover_topology"], attempts=[])
    from temsim.simulation_modes import mode_key as simulation_mode_key
    report['simulation_mode'] = simulation_mode_key(candidate)
    measurements, failures = {}, {}
    evaluations = 0

    class BudgetExhausted(RuntimeError):
        pass

    def measure(values):
        nonlocal evaluations
        identity = tuple(map(float, values))
        if identity in failures:
            raise ValueError(failures[identity])
        if identity not in measurements:
            if evaluations >= maximum_evaluations:
                raise BudgetExhausted("Full-trajectory evaluation budget exhausted")
            for k, v in zip(keys, values):
                lenses[k].percent = float(v)
            evaluations += 1
            try:
                measurements[identity] = checkpoint.measure(values)
            except (ValueError, FloatingPointError) as error:
                failures[identity] = str(error)
                raise
            m = measurements[identity]
            progress(f"{mode_key}: {evaluations} traces; alpha95={m.statistics.convergence_95_mrad:.6g} mrad; "
                     f"D95={m.statistics.illumination_diameter_95_um:.6g} um; waist={m.local_waist_offset_nm:.6g} nm")
        return measurements[identity]

    with threadpool_limits(1):
        from temsim.optics.illumination_checkpoint import IlluminationCheckpoint
        progress(f"{mode_key}: executing the physical gun and fixed upstream column")
        checkpoint = IlluminationCheckpoint(candidate, keys, step_mm=.05)
        report["executed_prefix_z_mm"] = checkpoint.prefix_z_mm
        nested = keys[-1] == 'objective_lens' and mode_key == 'nano_probe'
        proposals = ([initial] if keys[-1] == 'objective_lens'
            and not search_branches else _proposals(candidate, keys, target, initial, maximum_evaluations,
                                                    neighbouring_first=not search_branches))
        best = None
        def rank(m):
            return (not (target.accepts(m) and diameter_gate(candidate, m, mode_key)['passed']),
                    float(np.linalg.norm(target.residual(m.statistics))))
        for seed in proposals:
            try:
                if nested:
                    vector = _focus_branch(measure, target, seed, upper)
                    optimizer_success = True
                else:
                    fit = least_squares(lambda v: target.residual(measure(v).statistics), seed,
                        bounds=(np.zeros(2), upper), max_nfev=maximum_evaluations, diff_step=1e-6,
                        x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10)
                    vector, optimizer_success = fit.x, bool(fit.success)
                m = measure(vector)
                score = float(np.linalg.norm(target.residual(m.statistics)))
                report["attempts"].append(dict(values=vector.tolist(), score=score,
                    optimizer_success=optimizer_success, measurement=asdict(m)))
                if best is None or rank(m) < best[0]:
                    best = (rank(m), vector.copy(), m)
                if target.accepts(m) and diameter_gate(candidate, m, mode_key)['passed']:
                    break
            except (ValueError, FloatingPointError) as error:
                report["attempts"].append(dict(error=str(error)))
            except BudgetExhausted as error:
                report["attempts"].append(dict(error=str(error)))
                break
        # Preserve the best executed point even if SciPy was interrupted by
        # the shared budget (Jacobian probes count as real traces too).
        for values, m in measurements.items():
            try:
                score = rank(m)
                if best is None or score < best[0]:
                    best = (score, np.asarray(values), m)
            except ValueError:
                continue
        report['evaluations'] = evaluations
        report['evaluation_failures'] = [dict(values=list(v), error=error) for v, error in failures.items()]
        if best is None:
            return None, report
        _, vector, coarse = best
        for k, v in zip(keys, vector):
            lenses[k].percent = float(v)
        fine = measure_surface_focus(candidate, step_mm=.025)
    report['diameter_gate'] = diameter_gate(candidate, fine, mode_key)
    report['pending_gates'].remove('probe_diameter')
    report['failed_gates'] = [] if report['diameter_gate']['passed'] else ['probe_diameter']
    report['aperture_gate'] = aperture_gate(candidate)
    try:
        report['flux_selection'] = set_calibration_flux(candidate, fine)
    except ValueError as error:
        report['flux_selection_error'] = str(error)
    report['current_gate'] = current_gate(candidate, fine)
    if not report['current_gate']['passed']:
        report['failed_gates'].append('specimen_current')
    report['column_current_limit_percent'] = candidate.column_current_limit_percent
    report['executed_input_digest'] = capture_instrument_snapshot(candidate).physical_digest
    report.update(strengths={l.key: l.percent for l in candidate.lenses if l.enabled
                  and l.z_mm <= candidate.sample.z_mm}, coarse=asdict(coarse), fine=asdict(fine),
                  polarities={l.key: l.polarity for l in candidate.lenses if l.enabled
                              and l.z_mm <= candidate.sample.z_mm},
                  c2_aperture_diameter_mm=candidate.condenser_aperture_2.diameter_mm,
                  focus_angle_pass=target.accepts(coarse) and target.accepts(fine))
    if not report['focus_angle_pass']:
        report['failed_gates'].append('illumination_target')
    return candidate, report
