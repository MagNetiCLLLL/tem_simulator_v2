from pathlib import Path
p=Path("src/temsim/optics/direct_alignment.py")
s=p.read_text()
a=s.index("def _refine_nanoprobe_production_focus(")
b=s.index("\ndef _validate_projector_production(",a)
s=s[:a]+'''def _refine_nanoprobe_production_focus(
    state,
    definition: DirectAlignmentDefinition,
    target: float,
    vector: np.ndarray,
    step_mm: float,
) -> tuple[np.ndarray, int]:
    """Polish the production focus without discarding an acceptable pupil.

    A clipped current-weighted angular quantile is not a smooth lens response.
    When the first-order candidate already has an acceptable convergence,
    preserve C2 and solve the local C3 focus first.  A coupled Newton step is
    only a fallback, and must not replace a better production-validated point.
    The final Direct Alignment angle, waist and step-spread gates still apply.
    """
    if definition.key != NANOPROBE_CONVERGENCE:
        return np.asarray(vector, dtype=float), 0
    initial = np.asarray(vector, dtype=float).copy()
    lenses = _lens_map(state)
    upper = np.asarray(
        [float(lenses[key].max_percent) for key in CONDENSER_KEYS], dtype=float
    )
    angle_tolerance = _target_number(definition, "maximum_relative_error", 0.03)
    waist_tolerance = _target_number(definition, "maximum_waist_offset_mm", 0.002)
    measurements: dict[tuple[float, ...], DirectAlignmentMeasurement] = {}

    def measure(candidate):
        key = tuple(float(value) for value in candidate)
        if key not in measurements:
            measurements[key] = _validate_condenser_production(
                state, definition, candidate, step_mm
            )
        return measurements[key]

    def angle_error(measured):
        return abs(math.log(max(measured.value, 1.0e-15) / float(target)))

    def acceptable(measured):
        return (
            angle_error(measured) <= angle_tolerance
            and abs(measured.constraint_value) <= waist_tolerance
        )

    def score(candidate):
        measured = measure(candidate)
        # Every valid point ranks ahead of every invalid one.
        return max(
            angle_error(measured) / angle_tolerance,
            abs(measured.constraint_value) / waist_tolerance,
        )

    def polish_c3(vector):
        candidate = np.asarray(vector, dtype=float).copy()

        def waist_at_c3(percent):
            probe = candidate.copy()
            probe[1] = float(percent)
            return measure(probe).constraint_value

        centre = float(candidate[1])
        if abs(waist_at_c3(centre)) <= 1.0e-12:
            return candidate
        for half_width in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
            lower = max(0.0, centre - half_width)
            higher = min(float(upper[1]), centre + half_width)
            lower_waist, upper_waist = waist_at_c3(lower), waist_at_c3(higher)
            if lower_waist == 0.0:
                candidate[1] = lower
                return candidate
            if upper_waist == 0.0:
                candidate[1] = higher
                return candidate
            if lower_waist * upper_waist < 0.0:
                candidate[1] = brentq(
                    waist_at_c3, lower, higher, xtol=1.0e-7, rtol=1.0e-10
                )
                return candidate
        return candidate

    base = measure(initial)
    if acceptable(base):
        return initial, len(measurements)
    candidates = [initial]
    if angle_error(base) <= angle_tolerance:
        focused = polish_c3(initial)
        candidates.append(focused)
        if acceptable(measure(focused)):
            return focused, len(measurements)

    residual = np.asarray((
        math.log(max(base.value, 1.0e-15) / float(target)),
        base.constraint_value,
    ))
    perturbation = 1.0e-3
    jacobian = np.empty((2, 2), dtype=float)
    for index in range(2):
        shifted = initial.copy()
        shifted[index] = min(shifted[index] + perturbation, upper[index])
        actual_step = shifted[index] - initial[index]
        if actual_step <= 0.0:
            return min(candidates, key=score), len(measurements)
        measured = measure(shifted)
        shifted_residual = np.asarray((
            math.log(max(measured.value, 1.0e-15) / float(target)),
            measured.constraint_value,
        ))
        jacobian[:, index] = (shifted_residual - residual) / actual_step
    try:
        correction = np.linalg.solve(jacobian, -residual)
    except np.linalg.LinAlgError:
        return min(candidates, key=score), len(measurements)
    if not np.all(np.isfinite(correction)):
        return min(candidates, key=score), len(measurements)
    correction = np.clip(correction, -2.0, 2.0)
    candidate = np.clip(initial + correction, 0.0, upper)
    candidates.append(candidate)
    candidates.append(polish_c3(candidate))
    best = min(candidates, key=score)
    return best, len(measurements)

'''+s[b:]
p.write_text(s,encoding="utf-8")
