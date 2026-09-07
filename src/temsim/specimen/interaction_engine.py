"""Unified orchestration boundary for specimen-interaction solvers.

The engine preserves the validated numerical solvers while making their
accounting relationship explicit.  Particle-resolved elastic collisions are
copied into a common event ledger; aggregate inelastic channels remain
probability distributions, and characteristic X-rays remain a derived
relaxation observable rather than a second electron-loss population.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np

from temsim.specimen.interaction_types import (
    ConservationCheck,
    InteractionEvent,
    InteractionProcess,
    SpecimenInteractionRequest,
    SpecimenInteractionResult,
    SpecimenModelCoupling,
    SpecimenObservable,
)
from temsim.specimen.scene import SpecimenScene


def _inelastic_conservation(distribution) -> ConservationCheck:
    channels = tuple(
        (str(channel.key), float(channel.probability))
        for channel in distribution.channels
    ) + (("absorbed", float(distribution.absorbed_probability)),)
    return ConservationCheck(
        name="specimen_inelastic_probability",
        quantity="electron population probability",
        unit="1",
        input_value=1.0,
        output_channels=channels,
        absolute_tolerance=2.0e-12,
    )


def _elastic_conservation(transport) -> ConservationCheck | None:
    values = transport.metrics.get("outcome_weight_fractions")
    if not isinstance(values, dict) or not values:
        return None
    return ConservationCheck(
        name="specimen_elastic_terminal_weight",
        quantity="conditional electron weight",
        unit="1",
        input_value=1.0,
        output_channels=tuple(
            (str(key), float(value))
            for key, value in sorted(values.items())
        ),
        absolute_tolerance=2.0e-12,
    )


def _inelastic_energy_conservation(distribution) -> ConservationCheck | None:
    """Partition mean beam energy without claiming local energy deposition."""

    energy_ev = float(getattr(distribution, "beam_energy_kev", math.nan)) * 1.0e3
    if not math.isfinite(energy_ev) or energy_ev <= 0.0:
        return None
    outputs: list[tuple[str, float]] = []
    for channel in distribution.channels:
        probability = float(channel.probability)
        loss_ev = float(channel.energy_loss_ev)
        if (
            not math.isfinite(loss_ev)
            or loss_ev < 0.0
            or loss_ev > energy_ev
        ):
            raise ValueError("Inelastic channel energy loss is outside the beam energy")
        outputs.extend(
            (
                (
                    f"{channel.key}:tracked_kinetic",
                    probability * (energy_ev - loss_ev),
                ),
                (
                    f"{channel.key}:represented_transfer",
                    probability * loss_ev,
                ),
            )
        )
    outputs.append(
        (
            "untracked_absorption_sink",
            float(distribution.absorbed_probability) * energy_ev,
        )
    )
    return ConservationCheck(
        name="specimen_inelastic_mean_energy_partition",
        quantity="mean beam energy accounting per source electron",
        unit="eV/electron",
        input_value=energy_ev,
        output_channels=tuple(outputs),
        absolute_tolerance=max(2.0e-9, energy_ev * 2.0e-12),
    )


def _elastic_energy_conservation(transport) -> ConservationCheck | None:
    """Check constant-energy elastic histories across every terminal outcome."""

    terminal = getattr(transport, "terminal_electrons", None)
    input_energy = transport.metrics.get("incident_mean_energy_ev")
    if terminal is None or input_energy is None:
        return None
    energies = np.asarray(terminal.kinetic_energy_ev, dtype=float)
    weights = np.asarray(terminal.weight, dtype=float)
    outcomes = np.asarray(terminal.outcome, dtype=object)
    if energies.shape != weights.shape or outcomes.shape != weights.shape:
        raise ValueError("Elastic terminal energy arrays do not share one shape")
    output_channels = tuple(
        (
            f"{outcome}:terminal_kinetic",
            float(np.sum(weights[outcomes == outcome] * energies[outcomes == outcome])),
        )
        for outcome in sorted({str(value) for value in outcomes})
    )
    value = float(input_energy)
    return ConservationCheck(
        name="specimen_elastic_mean_energy",
        quantity="mean electron kinetic energy",
        unit="eV/electron",
        input_value=value,
        output_channels=output_channels,
        absolute_tolerance=max(2.0e-9, value * 2.0e-12),
    )


def _eds_relaxation_conservation(spectrum) -> ConservationCheck | None:
    """Check the direct vacancy relaxation partition without cascade claims."""

    vacancies = tuple(getattr(spectrum, "vacancies", ()))
    if not vacancies:
        return None
    vacancy_count = float(
        sum(row.expected_vacancies for row in vacancies)
    )
    return ConservationCheck(
        name="eds_direct_vacancy_relaxation",
        quantity="expected direct vacancy relaxations",
        unit="events/acquisition",
        input_value=vacancy_count,
        output_channels=(
            (
                "radiative",
                float(
                    sum(
                        row.expected_vacancies * row.fluorescence_yield
                        for row in vacancies
                    )
                ),
            ),
            (
                "auger",
                float(
                    sum(
                        row.expected_vacancies * row.auger_yield
                        for row in vacancies
                    )
                ),
            ),
            (
                "unresolved_shell_transfer",
                float(
                    sum(
                        row.expected_vacancies
                        * row.unresolved_relaxation_yield
                        for row in vacancies
                    )
                ),
            ),
        ),
        absolute_tolerance=max(2.0e-12, vacancy_count * 2.0e-12),
    )


def _elastic_event_ledger(transport) -> tuple[InteractionEvent, ...]:
    """Copy only genuinely position-resolved Monte Carlo collisions."""

    if transport is None:
        return ()
    model = str(
        transport.metrics.get("electron_transport_model", "elastic_transport")
    )
    rows: list[InteractionEvent] = []
    for trajectory_index, trajectory in enumerate(
        getattr(transport, "trajectories", ())
    ):
        for event_index, event in enumerate(trajectory.events):
            theta = float(event.theta_rad)
            phi = float(event.phi_rad)
            rows.append(
                InteractionEvent(
                    process=InteractionProcess.ELASTIC_SCATTER,
                    position_nm=tuple(float(value) for value in event.position_nm),
                    electron_weight=float(trajectory.incident_weight),
                    incident_energy_ev=float(trajectory.initial_energy_ev),
                    energy_transfer_ev=0.0,
                    event_id=(
                        f"elastic:trajectory{trajectory_index}:"
                        f"ray{trajectory.source_ray_index}:event{event_index}"
                    ),
                    parent_electron_index=int(trajectory.source_ray_index),
                    deflection_rad=(
                        theta * math.cos(phi),
                        theta * math.sin(phi),
                    ),
                    source_key=str(event.source_key),
                    material_key=str(event.material_key),
                    atomic_number=int(event.atomic_number),
                    model=model,
                    provenance=(
                        "stored elastic Monte Carlo collision; local "
                        "tangent-plane angular components"
                    ),
                )
            )
    return tuple(rows)


def _stable_fraction(identifier: str) -> float:
    """Map a stable event ID to an open unit interval reproducibly."""

    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (integer + 0.5) / float(1 << 64)


def _vacancy_position_nm(
    vacancy, material_flights
) -> tuple[float, float, float] | None:
    """Sample one point from already stored material flights without transport."""

    ray_index = getattr(vacancy, "source_ray_index", None)
    if ray_index is None:
        return None
    matches = []
    lengths = []
    for flight in material_flights:
        if (
            int(flight.source_ray_index) != int(ray_index)
            or str(flight.source_key) != str(vacancy.source_key)
            or str(flight.material_key) != str(vacancy.material_key)
            or str(flight.history) != str(vacancy.electron_history)
        ):
            continue
        start = np.asarray(flight.start_nm, dtype=float)
        end = np.asarray(flight.end_nm, dtype=float)
        length = float(np.linalg.norm(end - start))
        if length > 0.0:
            matches.append((start, end))
            lengths.append(length)
    total_length = float(sum(lengths))
    if total_length <= 0.0:
        return None
    target = _stable_fraction(str(vacancy.vacancy_id)) * total_length
    traversed = 0.0
    for (start, end), length in zip(matches, lengths, strict=True):
        if target <= traversed + length:
            fraction = (target - traversed) / length
            position = start + fraction * (end - start)
            return tuple(float(value) for value in position)
        traversed += length
    return tuple(float(value) for value in matches[-1][1])


def _eds_event_ledger(
    spectrum,
    transport,
) -> tuple[InteractionEvent, ...]:
    """Build linked vacancy/X-ray records from one EDS calculation pass.

    The shell cross sections and relaxation yields are never evaluated here.
    Position sampling only consumes the material flights already generated by
    the same elastic Monte Carlo run used to create the spectrum tracks.
    """

    if spectrum is None or transport is None:
        return ()
    material_flights = tuple(getattr(transport, "material_flights", ()))
    if not material_flights:
        return ()
    # Keep each group's original order and duplicates: deterministic vacancy
    # sampling is length-weighted over these exact flights.  Index references
    # once instead of searching every electron's flights for every vacancy.
    flights_by_source: dict[tuple[int, str, str, str], list[object]] = {}
    for flight in material_flights:
        flight_key = (
            int(flight.source_ray_index),
            str(flight.source_key),
            str(flight.material_key),
            str(flight.history),
        )
        flights_by_source.setdefault(flight_key, []).append(flight)
    lines_by_vacancy: dict[str, list[object]] = {}
    for line in getattr(spectrum, "lines", ()):
        lines_by_vacancy.setdefault(str(line.vacancy_id), []).append(line)
    spectrum_metrics = getattr(spectrum, "metrics", {})
    incident_electrons = float(
        spectrum_metrics.get("incident_electrons", 0.0)
    )
    source_electrons = float(
        spectrum_metrics.get(
            "source_electrons_before_column_losses", incident_electrons
        )
    )
    rows: list[InteractionEvent] = []
    for vacancy in getattr(spectrum, "vacancies", ()):
        ray_index = getattr(vacancy, "source_ray_index", None)
        matching_flights = (
            () if ray_index is None else flights_by_source.get(
                (int(ray_index), str(vacancy.source_key),
                 str(vacancy.material_key), str(vacancy.electron_history)),
                (),
            )
        )
        position = _vacancy_position_nm(vacancy, matching_flights)
        if position is None:
            continue
        vacancy_id = str(vacancy.vacancy_id)
        occurrences = (
            float(vacancy.expected_occurrences_per_incident_electron)
            * incident_electrons
            / source_electrons
            if source_electrons > 0.0
            else 0.0
        )
        rows.append(
            InteractionEvent(
                process=InteractionProcess.CORE_IONISATION,
                position_nm=position,
                electron_weight=float(vacancy.electron_weight),
                incident_energy_ev=float(vacancy.electron_energy_ev),
                energy_transfer_ev=float(vacancy.edge_energy_ev),
                expected_occurrences_per_source_electron=occurrences,
                event_id=vacancy_id,
                parent_electron_index=int(vacancy.source_ray_index),
                source_key=str(vacancy.source_key),
                material_key=str(vacancy.material_key),
                atomic_number=int(vacancy.atomic_number),
                model="Bote-Salvat shell ionisation",
                provenance=(
                    "weighted vacancy sampled along an existing stored "
                    "elastic material flight; transfer is the shell-binding "
                    "lower bound and is excluded from the exclusive "
                    "inelastic population budget"
                ),
            )
        )
        radiative_id = f"{vacancy_id}:radiative"
        if float(vacancy.fluorescence_yield) > 0.0:
            rows.append(
                InteractionEvent(
                    process=InteractionProcess.RADIATIVE_RELAXATION,
                    position_nm=position,
                    electron_weight=float(vacancy.electron_weight),
                    incident_energy_ev=float(vacancy.electron_energy_ev),
                    emitted_energy_ev=None,
                    expected_occurrences_per_source_electron=(
                        occurrences * float(vacancy.fluorescence_yield)
                    ),
                    event_id=radiative_id,
                    parent_event_id=vacancy_id,
                    parent_electron_index=int(vacancy.source_ray_index),
                    source_key=str(vacancy.source_key),
                    material_key=str(vacancy.material_key),
                    atomic_number=int(vacancy.atomic_number),
                    model="xraylib fluorescence yield",
                    provenance=(
                        "direct radiative branch; photon energy is resolved "
                        "only by its characteristic-line child records"
                    ),
                )
            )
        if source_electrons <= 0.0:
            continue
        for line_index, line in enumerate(lines_by_vacancy.get(vacancy_id, ())):
            rows.append(
                InteractionEvent(
                    process=InteractionProcess.CHARACTERISTIC_X_RAY,
                    position_nm=position,
                    electron_weight=float(vacancy.electron_weight),
                    incident_energy_ev=float(vacancy.electron_energy_ev),
                    emitted_energy_ev=float(line.energy_ev),
                    expected_occurrences_per_source_electron=(
                        float(line.expected_emitted_photons)
                        / source_electrons
                    ),
                    event_id=(
                        f"{vacancy_id}:xray{line_index}:"
                        f"{line.transition}"
                    ),
                    parent_event_id=radiative_id,
                    parent_electron_index=int(vacancy.source_ray_index),
                    source_key=str(vacancy.source_key),
                    material_key=str(vacancy.material_key),
                    atomic_number=int(vacancy.atomic_number),
                    transition=str(line.transition),
                    model="xraylib radiative relaxation",
                    provenance=(
                        "weighted characteristic photon derived from the "
                        "same stored vacancy contribution as the spectrum"
                    ),
                )
            )
        if float(vacancy.auger_yield) > 0.0:
            rows.append(
                InteractionEvent(
                    process=InteractionProcess.AUGER_RELAXATION,
                    position_nm=position,
                    electron_weight=float(vacancy.electron_weight),
                    incident_energy_ev=float(vacancy.electron_energy_ev),
                    emitted_energy_ev=None,
                    expected_occurrences_per_source_electron=(
                        occurrences * float(vacancy.auger_yield)
                    ),
                    event_id=f"{vacancy_id}:auger",
                    parent_event_id=vacancy_id,
                    parent_electron_index=int(vacancy.source_ray_index),
                    source_key=str(vacancy.source_key),
                    material_key=str(vacancy.material_key),
                    atomic_number=int(vacancy.atomic_number),
                    model="xraylib Auger yield",
                    provenance=(
                        "direct non-radiative branch; Auger kinetic energy "
                        "and direction are not assigned by the current model"
                    ),
                )
            )
        if float(vacancy.unresolved_relaxation_yield) > 0.0:
            rows.append(
                InteractionEvent(
                    process=InteractionProcess.UNRESOLVED_RELAXATION,
                    position_nm=position,
                    electron_weight=float(vacancy.electron_weight),
                    incident_energy_ev=float(vacancy.electron_energy_ev),
                    emitted_energy_ev=None,
                    expected_occurrences_per_source_electron=(
                        occurrences
                        * float(vacancy.unresolved_relaxation_yield)
                    ),
                    event_id=f"{vacancy_id}:unresolved",
                    parent_event_id=vacancy_id,
                    parent_electron_index=int(vacancy.source_ray_index),
                    source_key=str(vacancy.source_key),
                    material_key=str(vacancy.material_key),
                    atomic_number=int(vacancy.atomic_number),
                    model="explicit direct-relaxation remainder",
                    provenance=(
                        "unresolved shell-transfer/Coster-Kronig or database "
                        "coverage remainder; no particle energy is invented"
                    ),
                )
            )
    return tuple(rows)


def _model_couplings(
    *,
    wave_imaging,
    inelastic_distribution,
    elastic_transport,
    eds_spectrum,
) -> tuple[SpecimenModelCoupling, ...]:
    rows: list[SpecimenModelCoupling] = []
    if wave_imaging is not None:
        rows.append(
            SpecimenModelCoupling(
                solver="wave_multislice",
                processes=(InteractionProcess.COHERENT_ELASTIC,),
                representation="coherent complex amplitude and intensity",
                event_resolution="not a classical per-collision history",
                energy_coupling="single-energy coherent propagation",
                contributes_to_exclusive_electron_budget=False,
                double_count_rule=(
                    "Do not add wave diffraction intensity to elastic-transport "
                    "terminal probabilities; they are parallel observables."
                ),
            )
        )
    if inelastic_distribution is not None:
        rows.append(
            SpecimenModelCoupling(
                solver="inelastic_poisson_branches",
                processes=(
                    InteractionProcess.PLASMON_LOSS,
                    InteractionProcess.CORE_IONISATION,
                    InteractionProcess.OTHER_INELASTIC,
                    InteractionProcess.ABSORPTION,
                ),
                representation="exclusive probability and mean-loss quadrature",
                event_resolution="aggregate channels without event positions",
                energy_coupling="branch energy offsets enter downstream optics",
                contributes_to_exclusive_electron_budget=True,
                double_count_rule=(
                    "This solver alone owns the additive electron population "
                    "budget; EDS photons are derived from ionisation."
                ),
            )
        )
    if elastic_transport is not None:
        rows.append(
            SpecimenModelCoupling(
                solver="elastic_particle_transport",
                processes=(InteractionProcess.ELASTIC_SCATTER,),
                representation="weighted finite-geometry particle trajectories",
                event_resolution="stored collisions have local 3-D positions",
                energy_coupling="elastic collisions retain kinetic energy",
                contributes_to_exclusive_electron_budget=False,
                double_count_rule=(
                    "Its normalized terminal outcomes are checked separately "
                    "and are not added to coherent-wave intensity."
                ),
            )
        )
    if eds_spectrum is not None:
        rows.append(
            SpecimenModelCoupling(
                solver="eds_shell_ionisation_and_radiative_relaxation",
                processes=(
                    InteractionProcess.CORE_IONISATION,
                    InteractionProcess.RADIATIVE_RELAXATION,
                    InteractionProcess.CHARACTERISTIC_X_RAY,
                    InteractionProcess.AUGER_RELAXATION,
                    InteractionProcess.UNRESOLVED_RELAXATION,
                ),
                representation=(
                    "expected shell vacancies, conserved direct relaxation "
                    "branches and detected photons"
                ),
                event_resolution=(
                    "path-integrated vacancy rates with linked representative "
                    "vacancy/X-ray event IDs on stored elastic flights"
                ),
                energy_coupling=(
                    "cross sections use track energy; X-ray line energies are "
                    "resolved, while Auger/cascade energies remain explicit "
                    "unknowns with no electron recoil feedback"
                ),
                contributes_to_exclusive_electron_budget=False,
                double_count_rule=(
                    "Characteristic photons are a derived child of core "
                    "ionisation and never add a second electron-loss probability; "
                    "the event ledger reuses rather than recomputes shell rates."
                ),
            )
        )
    return tuple(rows)


def _validated_conservation(
    checks: Iterable[ConservationCheck | None],
) -> tuple[ConservationCheck, ...]:
    result = tuple(check for check in checks if check is not None)
    for check in result:
        check.require_conserved()
    return result


_POINT_OBSERVABLES = frozenset(
    (
        SpecimenObservable.ELASTIC_TRANSPORT,
        SpecimenObservable.CHARACTERISTIC_X_RAY,
    )
)


def _same_optional_scalar(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return float(left) == float(right)


def _same_point_request(
    left: SpecimenInteractionRequest,
    right: SpecimenInteractionRequest,
) -> bool:
    """Return whether two requests address the same specimen acquisition.

    Reuse is intentionally exact: these values originate from the same GUI
    state, so a changed coordinate, dwell time or source count must invalidate
    the point result rather than being hidden by a numerical tolerance.
    """

    return all(
        _same_optional_scalar(left_value, right_value)
        for left_value, right_value in (
            (left.point_x_nm, right.point_x_nm),
            (left.point_y_nm, right.point_y_nm),
            (left.dwell_time_s, right.dwell_time_s),
            (left.incident_electrons, right.incident_electrons),
        )
    )


def _merged_request(
    existing: SpecimenInteractionResult | None,
    request: SpecimenInteractionRequest,
    *,
    preserve_existing_point: bool,
) -> SpecimenInteractionRequest:
    existing_observables = (
        set(existing.request.observables) if existing is not None else set()
    )
    if not preserve_existing_point:
        existing_observables.difference_update(_POINT_OBSERVABLES)
    observables = frozenset(existing_observables | set(request.observables))
    request_has_point_observable = bool(request.observables & _POINT_OBSERVABLES)
    point_source = (
        request
        if request_has_point_observable or existing is None
        else existing.request
    )
    return SpecimenInteractionRequest(
        observables=observables,
        point_x_nm=point_source.point_x_nm,
        point_y_nm=point_source.point_y_nm,
        dwell_time_s=point_source.dwell_time_s,
        incident_electrons=point_source.incident_electrons,
    )


def run_specimen_interactions(
    state,
    simulation,
    request: SpecimenInteractionRequest,
    *,
    detector_geometry=None,
    existing_result: SpecimenInteractionResult | None = None,
    progress_callback=None,
) -> SpecimenInteractionResult:
    """Run only explicitly requested specimen observables.

    An inelastic distribution already used by the global ray calculation is
    attached rather than recomputed.  Point EDS remains explicit and retains
    its selected straight-primary or elastic-Monte-Carlo transport mode.

    ``existing_result`` may only come from the same normalised state and global
    column result.  Compatible completed observables are retained, and only
    missing work is calculated.  Point observables are discarded together when
    their coordinate, dwell time or incident-electron request changes.
    """

    if existing_result is not None and not isinstance(
        existing_result, SpecimenInteractionResult
    ):
        raise TypeError("Existing specimen result has the wrong contract type")

    scene = (
        SpecimenScene.from_state(state)
        if getattr(state, "sample", None) is not None
        else None
    )
    dependency_signatures: dict[str, str] = {}
    if callable(getattr(state, "to_dict", None)):
        from temsim.calculation_cache import (
            calculation_signatures,
            matching_products,
        )
        from temsim.specimen.interaction_types import (
            retain_specimen_observables,
        )

        dependency_signatures = calculation_signatures(state)
        old_signatures = (
            existing_result.metrics.get("dependency_signatures", {})
            if existing_result is not None
            else {}
        )
        if existing_result is not None and old_signatures:
            reusable_products = matching_products(
                old_signatures, dependency_signatures
            )
            retained_observables: set[SpecimenObservable] = set()
            has_projector_checkpoint = bool(
                existing_result.wave_imaging is not None
                and getattr(
                    existing_result.wave_imaging,
                    "projector_checkpoint",
                    None,
                )
                is not None
            )
            if (
                "wave" in reusable_products
                or (
                    "wave_source" in reusable_products
                    and has_projector_checkpoint
                )
            ):
                retained_observables.add(
                    SpecimenObservable.COHERENT_ELASTIC_WAVE
                )
            if "elastic" in reusable_products:
                retained_observables.add(SpecimenObservable.ELASTIC_TRANSPORT)
            if "eds" in reusable_products:
                retained_observables.add(
                    SpecimenObservable.CHARACTERISTIC_X_RAY
                )
            if "incident" in reusable_products:
                retained_observables.add(
                    SpecimenObservable.STOCHASTIC_INELASTIC
                )
            existing_result = retain_specimen_observables(
                existing_result,
                frozenset(retained_observables),
            )
    compatible_existing = (
        existing_result
        if existing_result is not None and existing_result.scene == scene
        else None
    )
    requested = request.observables
    point_requested = bool(requested & _POINT_OBSERVABLES)
    preserve_existing_point = bool(
        compatible_existing is not None
        and (
            not point_requested
            or _same_point_request(compatible_existing.request, request)
        )
    )
    merged_request = _merged_request(
        compatible_existing,
        request,
        preserve_existing_point=preserve_existing_point,
    )
    reused: set[SpecimenObservable] = set()
    calculated: set[SpecimenObservable] = set()

    wave_imaging = (
        compatible_existing.wave_imaging
        if compatible_existing is not None
        else None
    )
    if wave_imaging is not None:
        reused.add(SpecimenObservable.COHERENT_ELASTIC_WAVE)

    if preserve_existing_point and compatible_existing is not None:
        eds_spectrum = compatible_existing.eds_spectrum
        elastic_transport = compatible_existing.elastic_transport
        incident_bundle = compatible_existing.incident_bundle
        if eds_spectrum is not None:
            reused.add(SpecimenObservable.CHARACTERISTIC_X_RAY)
        if elastic_transport is not None:
            reused.add(SpecimenObservable.ELASTIC_TRANSPORT)
    else:
        eds_spectrum = None
        elastic_transport = None
        incident_bundle = None

    inelastic_distribution = getattr(simulation, "real_interactions", None)
    if (
        inelastic_distribution is None
        and compatible_existing is not None
    ):
        inelastic_distribution = compatible_existing.inelastic_distribution
        if inelastic_distribution is not None:
            reused.add(SpecimenObservable.STOCHASTIC_INELASTIC)
    if SpecimenObservable.STOCHASTIC_INELASTIC in requested:
        if inelastic_distribution is None:
            from temsim.specimen.inelastic import real_inelastic_distribution

            inelastic_distribution = real_inelastic_distribution(state)
            calculated.add(SpecimenObservable.STOCHASTIC_INELASTIC)

    if (
        SpecimenObservable.COHERENT_ELASTIC_WAVE in requested
        and wave_imaging is None
    ):
        from temsim.physics.wave_imaging import simulate_wave_image

        wave_imaging = simulate_wave_image(state, simulation)
        calculated.add(SpecimenObservable.COHERENT_ELASTIC_WAVE)

    if (
        SpecimenObservable.CHARACTERISTIC_X_RAY in requested
        and eds_spectrum is None
    ):
        elastic_available_before_eds = elastic_transport is not None
        if detector_geometry is None:
            raise ValueError(
                "Characteristic X-ray calculation requires EDS detector geometry"
            )
        from temsim.detector.eds_signal import simulate_eds_point

        eds_kwargs = {
            "simulation": simulation,
            "x_nm": request.point_x_nm,
            "y_nm": request.point_y_nm,
            "dwell_time_s": request.dwell_time_s,
            "incident_electrons": request.incident_electrons,
        }
        if elastic_transport is not None:
            eds_kwargs["elastic_transport"] = elastic_transport
        if incident_bundle is not None:
            eds_kwargs["incident_bundle"] = incident_bundle
        if progress_callback is not None:
            eds_kwargs["progress_callback"] = progress_callback
        eds_spectrum = simulate_eds_point(
            state,
            detector_geometry,
            **eds_kwargs,
        )
        calculated.add(SpecimenObservable.CHARACTERISTIC_X_RAY)
        elastic_transport = eds_spectrum.elastic_transport
        if elastic_transport is not None:
            if elastic_available_before_eds:
                reused.add(SpecimenObservable.ELASTIC_TRANSPORT)
            else:
                calculated.add(SpecimenObservable.ELASTIC_TRANSPORT)

    if (
        SpecimenObservable.ELASTIC_TRANSPORT in requested
        and elastic_transport is None
    ):
        from temsim.specimen.elastic_transport import (
            incident_rays_from_simulation,
            simulate_elastic_point_transport,
        )

        incident_bundle = incident_rays_from_simulation(
            state,
            simulation,
            target_x_nm=request.point_x_nm,
            target_y_nm=request.point_y_nm,
        )
        transport_kwargs = {"incident_rays": incident_bundle.rays}
        if progress_callback is not None:
            transport_kwargs["progress_callback"] = (
                lambda done, total, label: progress_callback(
                    round(9900 * min(max(done, 0), total) / max(total, 1)),
                    10000, label,
                )
            ) if callable(progress_callback) else progress_callback
        elastic_transport = simulate_elastic_point_transport(
            state,
            **transport_kwargs,
        )
        calculated.add(SpecimenObservable.ELASTIC_TRANSPORT)

    spectrum_metrics = getattr(eds_spectrum, "metrics", None)
    if isinstance(spectrum_metrics, dict):
        spectrum_metrics.update(
            {
                "core_ionisation_accounting": (
                    "derived relaxation observable; not added to the "
                    "exclusive electron-loss probability budget"
                ),
                "inelastic_event_coupling": (
                    "vacancies and X-rays share event IDs within EDS; the "
                    "aggregate exclusive inelastic branches remain a "
                    "separate non-additive population model"
                ),
                "event_ledger_recomputes_shell_cross_sections": False,
            }
        )

    completed: set[SpecimenObservable] = set()
    if wave_imaging is not None:
        completed.add(SpecimenObservable.COHERENT_ELASTIC_WAVE)
    if inelastic_distribution is not None:
        completed.add(SpecimenObservable.STOCHASTIC_INELASTIC)
    if eds_spectrum is not None:
        completed.add(SpecimenObservable.CHARACTERISTIC_X_RAY)
    if elastic_transport is not None:
        completed.add(SpecimenObservable.ELASTIC_TRANSPORT)

    if callable(progress_callback):
        progress_callback(
            9900, 10000, "Recording specimen interaction events",
        )
    elastic_events = _elastic_event_ledger(elastic_transport)
    eds_events = _eds_event_ledger(eds_spectrum, elastic_transport)
    events = elastic_events + eds_events
    couplings = _model_couplings(
        wave_imaging=wave_imaging,
        inelastic_distribution=inelastic_distribution,
        elastic_transport=elastic_transport,
        eds_spectrum=eds_spectrum,
    )
    conservation = _validated_conservation(
        (
            _inelastic_conservation(inelastic_distribution)
            if inelastic_distribution is not None
            else None,
            _inelastic_energy_conservation(inelastic_distribution)
            if inelastic_distribution is not None
            else None,
            _elastic_conservation(elastic_transport)
            if elastic_transport is not None
            else None,
            _elastic_energy_conservation(elastic_transport)
            if elastic_transport is not None
            else None,
            _eds_relaxation_conservation(eds_spectrum)
            if eds_spectrum is not None
            else None,
        )
    )
    metrics: dict[str, object] = {
        "contract_version": 5,
        "dependency_signatures": dependency_signatures,
        "coordinate_system": "right_handed_specimen_frame_electrons_along_+z",
        "position_unit": "nm",
        "direction_unit": "1",
        "angle_unit": "rad",
        "energy_unit": "eV",
        "time_unit": "s",
        "weight_unit": "1",
        "requested_observables": tuple(
            sorted(value.value for value in merged_request.observables)
        ),
        "completed_observables": tuple(
            sorted(value.value for value in completed)
        ),
        "solver_ownership": {
            "coherent_elastic_and_channeling": "wave_multislice",
            "exclusive_electron_energy_loss": "inelastic_poisson_branches",
            "finite_geometry_elastic": "elastic_particle_transport",
            "derived_characteristic_x_ray": "eds_signal",
        },
        "event_ledger_representation": (
            "stored_elastic_collisions_and_shared_weighted_eds_events"
            if eds_events
            else "stored_elastic_monte_carlo_collisions_only"
            if elastic_events
            else "no_position_resolved_solver_requested"
        ),
        "per_event_ledger_populated": bool(events),
        "event_ledger_count": len(events),
        "event_ledger_weighted_collision_incidence": float(
            sum(event.electron_weight for event in elastic_events)
        ),
        "event_ledger_expected_vacancies_per_source_electron": float(
            sum(
                event.expected_occurrences_per_source_electron
                for event in eds_events
                if event.process is InteractionProcess.CORE_IONISATION
            )
        ),
        "event_ledger_expected_emitted_xrays_per_source_electron": float(
            sum(
                event.expected_occurrences_per_source_electron
                for event in eds_events
                if event.process is InteractionProcess.CHARACTERISTIC_X_RAY
            )
        ),
        "event_ledger_expected_radiative_relaxations_per_source_electron": float(
            sum(
                event.expected_occurrences_per_source_electron
                for event in eds_events
                if event.process is InteractionProcess.RADIATIVE_RELAXATION
            )
        ),
        "event_ledger_expected_auger_relaxations_per_source_electron": float(
            sum(
                event.expected_occurrences_per_source_electron
                for event in eds_events
                if event.process is InteractionProcess.AUGER_RELAXATION
            )
        ),
        "event_ledger_expected_unresolved_relaxations_per_source_electron": float(
            sum(
                event.expected_occurrences_per_source_electron
                for event in eds_events
                if event.process is InteractionProcess.UNRESOLVED_RELAXATION
            )
        ),
        "eds_event_ledger_recomputed_cross_sections": False,
        "eds_and_spectrum_share_vacancy_contributions": bool(eds_spectrum),
        "event_ledger_completeness": (
            "representative stored elastic trajectories; EDS vacancy and "
            "X-ray positions are sampled only on their existing stored "
            "material flights; aggregate inelastic branches remain unlocated"
            if eds_events
            else "representative stored elastic trajectories; aggregate "
            "inelastic and EDS channels have no fabricated positions"
            if elastic_events
            else "not applicable"
        ),
        "model_couplings": tuple(
            {
                "solver": coupling.solver,
                "processes": tuple(
                    process.value for process in coupling.processes
                ),
                "representation": coupling.representation,
                "event_resolution": coupling.event_resolution,
                "energy_coupling": coupling.energy_coupling,
                "contributes_to_exclusive_electron_budget": (
                    coupling.contributes_to_exclusive_electron_budget
                ),
                "double_count_rule": coupling.double_count_rule,
            }
            for coupling in couplings
        ),
        "exclusive_electron_budget_owner_count": sum(
            bool(coupling.contributes_to_exclusive_electron_budget)
            for coupling in couplings
        ),
        "existing_result_reused": bool(reused),
        "existing_result_invalidated_by_scene_change": bool(
            existing_result is not None and compatible_existing is None
        ),
        "reused_observables": tuple(
            sorted(value.value for value in reused)
        ),
        "calculated_observables_this_call": tuple(
            sorted(value.value for value in calculated)
        ),
        "shared_result_rule": (
            "reuse compatible outputs from the same state and column result; "
            "never recalculate an observable only for display"
        ),
        "conservation_checks": tuple(
            {
                "name": check.name,
                "quantity": check.quantity,
                "unit": check.unit,
                "input": check.input_value,
                "output": check.output_value,
                "residual": check.residual,
                "absolute_tolerance": check.absolute_tolerance,
                "conserved": check.conserved,
            }
            for check in conservation
        ),
    }
    if callable(progress_callback):
        progress_callback(10000, 10000, "Specimen interactions complete")
    return SpecimenInteractionResult(
        request=merged_request,
        completed_observables=frozenset(completed),
        scene=scene,
        incident_bundle=incident_bundle,
        wave_imaging=wave_imaging,
        inelastic_distribution=inelastic_distribution,
        elastic_transport=elastic_transport,
        eds_spectrum=eds_spectrum,
        events=events,
        couplings=couplings,
        conservation=conservation,
        metrics=metrics,
    )
