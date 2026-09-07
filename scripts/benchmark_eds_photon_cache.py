"""Compare exact photon geometry reuse on CPU; no electron propagation.

Run with the project environment from the repository root. The fixture uses
100 prescribed emission origins, not a new Monte Carlo or STEM acquisition.
"""

import json
import statistics
from time import perf_counter

from temsim.assembly_catalog import AssemblyCatalog
from temsim.component_keys import EDS_DETECTOR_SYSTEM
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.detector import eds_photon_transport as transport
from temsim.detector.eds_signal import _radiative_lines
from temsim.optics.column import default_state


def main():
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    geometry = EDSDetectorArrayGeometry.from_part_data(assembly.part(EDS_DETECTOR_SYSTEM).data)
    state.sample.specimen_mode = "virtual"
    state.sample.specimen_preset_key = "si_110"
    state.sample.thickness_nm = 10.0
    lines = _radiative_lines(14, 0)
    rays = []
    for index in range(100):
        origin = (
            (index % 10 - 4.5) * 0.02e-6,
            (index // 10 - 4.5) * 0.02e-6,
            float(state.sample.z_mm) + (index % 9 - 4) * 1e-6,
        )
        for transition, energy_ev, rate in lines:
            rays.extend(transport.detector_quadrature_photons(
                emission_key=f"vacancy{index}:{transition}", origin_mm=origin,
                energy_ev=energy_ev, expected_emitted_photons=(1 + index / 100) * rate,
                geometry=geometry, quadrature_order=1, transition=transition,
            ))
    rays = tuple(rays)
    assert len(lines) == 4 and geometry.segment_count == 6 and len(rays) == 2400
    original_limit = transport._PHOTON_GEOMETRY_CACHE_ENTRIES
    timings = {0: [], 1024: []}
    baseline = None
    try:
        # Warm both routes, then alternate ordering in three paired runs.
        for pair_index, order in enumerate(((0, 1024), (0, 1024), (1024, 0), (0, 1024))):
            for limit in order:
                transport._PHOTON_GEOMETRY_CACHE_ENTRIES = limit
                started = perf_counter()
                result = transport.transport_eds_photons(state, rays, geometry)
                elapsed = perf_counter() - started
                if baseline is None:
                    baseline = result
                assert result == baseline
                if pair_index:
                    timings[limit].append(elapsed)
                print(json.dumps({"warmup": pair_index == 0, "cache_entries": limit,
                                  "elapsed_s": elapsed, "exact_equal": True}), flush=True)
    finally:
        transport._PHOTON_GEOMETRY_CACHE_ENTRIES = original_limit
    print(json.dumps({
        "workload": "CPU-only photon transport, prebuilt photons, installed objective pole and specimen/support geometry",
        "origins": 100, "Si_K_transitions": len(lines), "directions": geometry.segment_count,
        "photons": len(rays), "unique_origin_direction_pairs": len({(ray.origin_mm, ray.direction) for ray in rays}),
        "pole_occluders": baseline.metrics["pole_occluder_count"],
        "finite_specimen_attenuation": baseline.metrics["finite_specimen_attenuation"],
        "support_material_attenuation": baseline.metrics["support_material_attenuation"],
        "detected_weight": baseline.metrics["total_detected_weight"],
        "warm_uncached_s": timings[0], "warm_cached_s": timings[1024],
        "median_uncached_s": statistics.median(timings[0]),
        "median_cached_s": statistics.median(timings[1024]),
        "exact_complete_result_equality": True,
    }, indent=2))


if __name__ == "__main__":
    main()
