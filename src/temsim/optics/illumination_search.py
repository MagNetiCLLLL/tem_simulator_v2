"""First-order focus-branch proposals, never accepted illumination presets.

The physical gun is executed normally. A round-lens transfer map then scans
objective-focus roots at fixed primary excitation. It retains real pupil
locations, but neglects nonlinear ray forces and column-wall clipping: every
proposal therefore needs full transport, sampling and topology validation.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq

from temsim.optics.assembly_illumination import TARGETS
from temsim.optics.surface_probe_focus import surface_z_mm
from temsim.physics.beam_statistics import transverse_beam_statistics


@dataclass(frozen=True)
class FocusProposal:
    values: tuple[float, float]
    alpha95_mrad: float
    diameter95_nm: float
    surviving_rays: int
    current_fraction: float
    score: float


def focus_grid_proposals(state, keys, mode, *, primary_points=49,
                         objective_points=65, keep=8, progress=lambda text: None):
    """Scan focus branches rather than optimize across narrow axial waists.

    Source, physical component strengths and geometry are not changed. Returns
    diverse finite candidates in excitation percent, plus numerical counters.
    """
    from temsim.optics.direct_alignment import _LiveFirstOrderModel
    from temsim.optics.electron_gun.source import trace_source_to_exit
    if mode not in TARGETS or len(keys) != 2 or keys[-1] != 'objective_lens':
        raise ValueError('Declare an illumination mode and primary/objective lens pair')
    if primary_points < 3 or objective_points < 3 or keep < 1:
        raise ValueError('Positive search dimensions with at least three points are required')
    target = TARGETS[mode]
    start, end = state.electron_gun.exit_plane_z_mm, surface_z_mm(state)
    e = trace_source_to_exit(state).exit_bundle
    source = np.asarray([e.x_m, e.y_m, e.tx_rad, e.ty_rad])
    apertures = [a for a in state.apertures if a.enabled and getattr(a, 'installed', True)
                 and start < a.z_mm < end]
    if state.nanopulser.installed:
        apertures.append(state.nanopulser.aperture)
    apertures.sort(key=lambda a: a.z_mm)
    planes = [a.z_mm for a in apertures]+[end]
    model = _LiveFirstOrderModel(state, start, end, keys, step_mm=.1, capture_z_mm=planes)
    if model.vector_maps:
        raise ValueError('First-order branch scouting is not qualified for mapped fields')
    evaluations, proposals = 0, []

    def measure(primary, objective):
        nonlocal evaluations
        evaluations += 1
        coordinates = model.matrices_at((primary, objective), planes) @ source
        alive = e.alive.copy() & (e.weight > 0)
        for aperture, rays in zip(apertures, coordinates[:-1]):
            if hasattr(aperture, 'transmission_mask'):
                alive &= aperture.transmission_mask(rays[0]*1e3, rays[1]*1e3)
            else:
                alive &= np.hypot(rays[0]*1e3-aperture.offset_x_mm,
                                  rays[1]*1e3-aperture.offset_y_mm) <= aperture.radius_mm
        if np.count_nonzero(alive) < 16:
            raise ValueError('Insufficient transmitted seed rays')
        return transverse_beam_statistics(*coordinates[-1], alive=alive, weights=e.weight)

    def correlation(primary, objective):
        s = measure(primary, objective)
        return s.radial_position_angle_covariance_m_rad / max(s.radius_rms_m*s.convergence_rms_rad, 1e-30)

    for index, primary in enumerate(np.linspace(.01, min(95., model.upper[0]-.01), primary_points)):
        previous = None
        for objective in np.linspace(.01, min(99., model.upper[1]-.01), objective_points):
            try:
                value = correlation(primary, objective)
                if previous is not None and previous[1]*value < 0:
                    root = brentq(lambda x: correlation(primary, x), previous[0], objective,
                                  xtol=1e-9, maxiter=60)
                    s = measure(primary, root)
                    # A zero covariance can also be a broad, nearly parallel
                    # state. Candidate ranking distinguishes that from a probe.
                    if mode == 'nano_probe':
                        score = abs(math.log(s.convergence_95_mrad/target.alpha95_mrad))
                        score += max(0., math.log(max(s.illumination_diameter_95_um*1e3, 1e-30)/10.))
                    else:
                        score = abs(math.log(s.illumination_diameter_95_um/target.diameter95_um))
                        score += max(0., math.log(s.convergence_95_mrad/target.alpha95_mrad))
                    proposals.append(FocusProposal((float(primary),float(root)), s.convergence_95_mrad,
                        s.illumination_diameter_95_um*1e3, s.surviving_rays, s.surviving_fraction, score))
                previous = (objective,value)
            except (ValueError, FloatingPointError):
                previous = None  # Never bridge a region with no resolved rays.
        if index % 12 == 0:
            progress(f'{keys[0]} focus scan {index+1}/{primary_points}; {len(proposals)} roots')
    selected = []
    for proposal in sorted(proposals, key=lambda p:p.score):
        if all(np.linalg.norm(np.asarray(proposal.values)-p.values) > 2. for p in selected):
            selected.append(proposal)
        if len(selected) >= keep:
            break
    return selected, dict(first_order_evaluations=evaluations, focus_roots=len(proposals),
                          qualification='PROPOSALS_ONLY')
