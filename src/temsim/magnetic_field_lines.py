"""Bounded display geometry for integral curves of a captured magnetic field.

Coordinates and distances are SI metres, fields are tesla.  The curves solve
``dr/ds = B / |B|`` in both directions; they are not electron trajectories.
Cross-section seed acceptance uses an explicit, fixed field reference and a
six-decade logarithmic display compression. Line density is consequently
qualitative, not a calibrated magnetic-flux measure. A weak transverse field
is not normalised independently of stronger axial fields in the same scene.
Nothing in this module emits particles, solves a field, or changes optics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class MagneticFieldScene(Protocol):
    bounds_m: np.ndarray
    seed_regions_m: tuple[np.ndarray, ...]
    source_keys: tuple[str, ...]
    notes: tuple[str, ...]

    def contains(self, points_m: np.ndarray) -> np.ndarray: ...

    def field_at_global_positions_t(self, points_m: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class MagneticFieldLines:
    segments_m: np.ndarray
    strengths_t: np.ndarray
    direction_segments_m: np.ndarray
    bounds_m: np.ndarray
    line_count: int
    seed_count: int
    reference_t: float
    notes: tuple[str, ...]


def field_strength_fraction(strengths_t, reference_t: float, *, gain: float = 1.0) -> np.ndarray:
    """Map total ``|B|`` to a fixed six-decade logarithmic display scale.

    Zero maps to zero and ``reference_t`` maps to one, with saturation above
    the reference. ``gain`` controls seed density only; colours normally use
    its default of one. Neither a scene maximum nor a component maximum is
    computed here, so a fixed reference remains comparable between scenes.
    This is a display transfer function, not a change to the field in tesla.
    """
    strengths = np.asarray(strengths_t, dtype=float)
    reference_t, gain = float(reference_t), float(gain)
    if not np.isfinite(reference_t) or reference_t <= 0:
        raise ValueError("reference_t must be finite and positive")
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("gain must be finite and nonnegative")
    if not np.all(np.isfinite(strengths)) or np.any(strengths < 0):
        raise ValueError("strengths_t must be finite and nonnegative")
    if gain == 0:
        return np.zeros_like(strengths)
    # Clipping the ratio before log1p also avoids overflow at extreme finite
    # display settings. Overflowing ratios saturate; no field is modified.
    with np.errstate(over="ignore", under="ignore"):
        ratio = np.minimum((strengths / reference_t) * gain, 1.0)
    return np.log1p(ratio * 1e6) / np.log1p(1e6)


def _readonly(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values.setflags(write=False)
    return values


def _box(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (2, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite (2, 3) array in metres")
    if np.any(result[1] <= result[0]):
        raise ValueError(f"{name} must have positive extent on every axis")
    return result


def _radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    """Reproducible low-discrepancy coordinates, independent of field strength."""
    digits = np.asarray(indices, dtype=np.int64).copy()
    values = np.zeros(len(digits), dtype=float)
    factor = 1.0 / base
    while np.any(digits):
        values += factor * (digits % base)
        digits //= base
        factor /= base
    return values


def _sample(scene: MagneticFieldScene, points: np.ndarray, bounds: np.ndarray):
    """Never request a provider value outside its declared validity domain."""
    valid = np.all(np.isfinite(points), axis=1)
    valid &= np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)
    if np.any(valid):
        indices = np.flatnonzero(valid)
        inside = np.asarray(scene.contains(points[indices]), dtype=bool)
        if inside.shape != (len(indices),):
            raise ValueError("Magnetic scene contains() must return one mask per point")
        valid[indices] &= inside
    values = np.zeros_like(points)
    if np.any(valid):
        sampled = np.asarray(scene.field_at_global_positions_t(points[valid]), dtype=float)
        if sampled.shape != (np.count_nonzero(valid), 3):
            raise ValueError("Magnetic scene must return one three-component B vector per point")
        values[valid] = sampled
    valid &= np.all(np.isfinite(values), axis=1)
    strengths = np.linalg.norm(np.where(np.isfinite(values), values, 0.0), axis=1)
    return values, strengths, valid


def _candidates(scene: MagneticFieldScene, bounds: np.ndarray, max_lines: int):
    """One plane per region, normal to the dominant local field component.

    This prevents a uniform axial field from receiving multiple seeds along
    the very same line, as a regular three-dimensional seed lattice would.
    Region-specific deterministic offsets avoid repeated transverse positions
    when neighbouring components share an axis.
    """
    regions = []
    seen = set()
    for raw in scene.seed_regions_m:
        region = _box(raw, "seed region")
        region = np.stack((np.maximum(region[0], bounds[0]), np.minimum(region[1], bounds[1])))
        if np.any(region[1] <= region[0]):
            continue
        key = tuple(region.ravel())
        if key not in seen:
            regions.append(region)
            seen.add(key)
    if not regions:
        return np.empty((0, 3)), np.empty(0), np.empty(0)
    if len(regions) > max_lines:
        indices = np.linspace(0, len(regions) - 1, max_lines, dtype=int)
        regions = [regions[i] for i in indices]

    centres = np.asarray([(r[0] + r[1]) * 0.5 for r in regions])
    centre_b, centre_norm, centre_valid = _sample(scene, centres, bounds)
    # A quadrupole can have B=0 exactly on axis. Probe a small symmetric set
    # to find the active orientation without assuming that a zero centre is
    # an empty component or introducing an artificial on-axis field.
    probes = []
    for r, centre in zip(regions, centres):
        offset = (r[1] - r[0]) * 0.25
        probes.extend(centre + sign * np.eye(3) * offset for sign in (-1, 1))
    probe_b, _, probe_valid = _sample(scene, np.concatenate(probes), bounds)
    probe_b = np.where(probe_valid[:, None], np.abs(probe_b), 0).reshape(len(regions), 6, 3)
    orientations = np.argmax(np.sum(probe_b, axis=1), axis=1)
    nonzero_centres = centre_valid & (centre_norm > 0)
    orientations[nonzero_centres] = np.argmax(np.abs(centre_b[nonzero_centres]), axis=1)

    # A split lens can have its field lobes away from the box centre (or
    # opposite lobes can cancel there). Select a strong section from fixed
    # geometric probes. Uniform amplitude changes leave this choice unchanged.
    # Ties prefer the centre, which keeps a uniform field seeded only once.
    offsets = np.array((0., -.1, .1, -.2, .2, -.3, .3, -.4, .4))
    section_probes = np.repeat(centres[:, None, :], len(offsets), axis=1)
    for index, (region, normal) in enumerate(zip(regions, orientations)):
        section_probes[index, :, normal] += offsets * (region[1, normal] - region[0, normal])
    _, section_strength, section_valid = _sample(scene, section_probes.reshape(-1, 3), bounds)
    section_strength = np.where(section_valid, section_strength, -1.).reshape(len(regions), -1)
    section_indices = np.argmax(section_strength, axis=1)
    section_centres = section_probes[np.arange(len(regions)), section_indices]

    seeds, steps, thresholds = [], [], []
    for index, (region, normal) in enumerate(zip(regions, orientations)):
        count = max_lines // len(regions) + (index < max_lines % len(regions))
        # Margins avoid exact boundary interpolation and repetitive corner seeds.
        sequence = np.arange(1, count + 1) + index * 104729
        u = 0.08 + 0.84 * _radical_inverse(sequence, 2)
        v = 0.08 + 0.84 * _radical_inverse(sequence, 3)
        axes = [axis for axis in range(3) if axis != normal]
        extent = region[1] - region[0]
        positions = np.tile(section_centres[index], (count, 1))
        positions[:, axes[0]] = region[0, axes[0]] + u * extent[axes[0]]
        positions[:, axes[1]] = region[0, axes[1]] + v * extent[axes[1]]
        seeds.append(positions)
        # Physical integration length, not an on-screen spacing. The smallest
        # box dimension resolves narrow supports and the stop masks between them.
        steps.append(np.full(count, float(np.min(extent)) * 0.20))
        thresholds.append(_radical_inverse(sequence, 5))
    return np.concatenate(seeds), np.concatenate(steps), np.concatenate(thresholds)


def build_field_lines(
    scene: MagneticFieldScene,
    *,
    reference_t: float,
    density: float = 1.0,
    max_lines: int = 640,
    max_steps: int = 64,
) -> MagneticFieldLines:
    """Trace a bounded, deterministic set of magnetic-field integral curves.

    ``max_steps`` is the maximum per direction per accepted seed. Storage is
    at most ``2 * max_lines * max_steps`` segments. All evaluations are batched
    NumPy operations; no worker or nested numerical-library pool is created.
    Invalid samples and weak fields terminate a line, never bridge a known gap.
    """
    reference_t = float(reference_t)
    density = float(density)
    if not np.isfinite(reference_t) or reference_t <= 0:
        raise ValueError("reference_t must be finite and positive")
    if not np.isfinite(density) or density < 0:
        raise ValueError("density must be finite and nonnegative")
    for value, name, ceiling in ((max_lines, "max_lines", 4096), (max_steps, "max_steps", 1024)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be an integer between 1 and {ceiling}")
    bounds = _box(scene.bounds_m, "bounds_m").copy()
    notes = list(scene.notes)
    notes.append("Magnetic field lines, not electron paths. Density is qualitative, not calibrated magnetic flux.")
    notes.append("Line density uses six-decade logarithmic compression of the total field; components share one reference.")
    notes.append(f"Fixed density reference: {reference_t:g} T; at most {max_lines} lines and {max_steps} steps per direction.")
    empty = np.empty((0, 2, 3), dtype=float)

    def result(segments=empty, strengths=np.empty(0), arrows=empty, lines=0, seeds=0):
        return MagneticFieldLines(
            _readonly(segments), _readonly(strengths), _readonly(arrows),
            _readonly(bounds), lines, seeds, reference_t, tuple(notes),
        )

    if density == 0:
        return result()
    candidates, candidate_steps, thresholds = _candidates(scene, bounds, max_lines)
    if not len(candidates):
        return result()
    _, seed_strength, valid = _sample(scene, candidates, bounds)
    # This threshold stays tied to an explicit reference, never each frame's
    # largest B. Uniform scaling therefore cannot silently renormalise density.
    weak_t = max(reference_t * 1e-9, np.finfo(float).tiny)
    # An approximately linear density mapping hides weak transverse correctors
    # beside a strong objective lens. Compress six decades of the total |B|
    # instead, with a smooth transition to zero and no independent source
    # scaling. Changing another component cannot alter this component's
    # acceptance merely by changing the strongest field in the scene.
    probability = field_strength_fraction(seed_strength, reference_t, gain=density)
    accepted = valid & (seed_strength > weak_t) & (thresholds < probability)
    seeds = candidates[accepted]
    step_size = candidate_steps[accepted]
    if not len(seeds):
        notes.append("No seeds exceed the fixed field-density threshold in the displayed domains.")
        return result()

    segments, strengths, arrows = [], [], []
    used = np.zeros(len(seeds), dtype=bool)
    exhausted = False
    for direction in (-1.0, 1.0):
        current = seeds.copy()
        active = np.ones(len(seeds), dtype=bool)
        for step in range(max_steps):
            indices = np.flatnonzero(active)
            if not len(indices):
                break
            start = current[indices]
            b0, n0, ok0 = _sample(scene, start, bounds)
            ok0 &= n0 > weak_t
            unit0 = np.divide(b0, n0[:, None], out=np.zeros_like(b0), where=n0[:, None] > weak_t)
            h = step_size[indices, None] * direction
            midpoint = start + 0.5 * h * unit0
            bm, nm, okm = _sample(scene, midpoint, bounds)
            okm &= nm > weak_t
            unitm = np.divide(bm, nm[:, None], out=np.zeros_like(bm), where=nm[:, None] > weak_t)
            end = start + h * unitm
            _, ne, oke = _sample(scene, end, bounds)
            oke &= ne > weak_t
            # Probe both chord quarter-points as well as the RK midpoint.
            # This stops across sampled map holes / separate provider domains
            # rather than connecting two valid endpoints through a known gap.
            quarters = start[:, None, :] + (end - start)[:, None, :] * np.array([0.25, 0.75])[None, :, None]
            _, nq, okq = _sample(scene, quarters.reshape(-1, 3), bounds)
            okq = (okq & (nq > weak_t)).reshape(-1, 2).all(axis=1)
            ok = ok0 & okm & oke & okq & (np.sum(unit0 * unitm, axis=1) > 0)
            active[indices[~ok]] = False
            if not np.any(ok):
                continue
            piece = np.stack((start[ok], end[ok]), axis=1)
            if direction < 0:
                piece = piece[:, ::-1, :]
            # Every segment is stored in +B order, including the backward arm.
            segments.append(piece)
            strengths.append(nm[ok])
            if step % 8 == 0:
                arrows.append(piece.copy())
            used[indices[ok]] = True
            current[indices[ok]] = end[ok]
        exhausted |= bool(np.any(active))
    if exhausted:
        notes.append("Some lines reached the display step budget; their ends are display limits.")
    if not segments:
        return result(seeds=len(seeds))
    return result(np.concatenate(segments), np.concatenate(strengths),
                  np.concatenate(arrows) if arrows else empty,
                  int(np.count_nonzero(used)), len(seeds))
