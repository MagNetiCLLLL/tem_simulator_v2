"""Build one structured result description from operations that actually ran."""
from copy import deepcopy

from temsim.excitation_calibration import calibration_from_recipe
from temsim.optics.aberration_basis import WAVE_TERMS, UNIMPLEMENTED_TERMS
from temsim.immutable_json import freeze_json, thaw_json


def attach_execution_evidence(result, state, product):
    metrics = result.metrics
    illumination = metrics.get("illumination_model", metrics.get("illumination_config", {}).get("model", "not_recorded"))
    records = metrics.get("illumination_executed_modes", ())
    modes = len(records) if records else (1 if hasattr(state,"_wave_source_node") else int(metrics.get("illumination_mode_count", 1)))
    phonons = int(metrics.get("specimen_configuration_count", 1))
    ledger = deepcopy(metrics.get("camera_flux_ledger", ()))
    nodes = [{"operation":"illumination", "model":illumination,"mode_count":modes,
              "scope":metrics.get("illumination_scope"),"modes":deepcopy(records)},
             {"operation":"specimen", "model":metrics.get("specimen_model"),
              "interaction_applied":metrics.get("specimen_sample_interaction_applied", metrics.get("sample_interaction_applied")),
              "configuration_count":phonons,"slice_count":metrics.get("specimen_slice_count"),
              "backend":metrics.get("specimen_compute_backend")}]
    nodes.extend({"operation":row["node_id"],**row} for row in ledger)
    if product == "STEM":
        nodes.append({"operation":"detector_reduction", "routing":metrics.get("physical_detector_routing"),
                      "bands_mrad":deepcopy(result.detector_ranges_mrad),"sampling":deepcopy(metrics.get("detector_sampling"))})
        if result.fourdstem_artifact is not None:
            nodes.append({"operation":"diffraction_capture","path":str(result.fourdstem_artifact.path),
                          "quantity":result.fourdstem_artifact.metadata["provenance"]["stored_frame_quantity"]})
    electrical = {}
    for lens in getattr(state,"lenses", ()):
        recipe = getattr(state,"lens_field_map_descriptors",{}).get(lens.key,{})
        if "ampere_turns" in recipe:
            electrical[lens.key] = calibration_from_recipe(recipe).at_control(lens.percent,getattr(lens,"polarity",1),enabled=lens.enabled)
    old = metrics.get("wave_execution_manifest",{})
    # Scientific validations are independent dimensions. No run-time setting,
    # cache hit or successful completion upgrades them to validated physics.
    verification = {"execution":"PASS", "numerical_convergence":"NOT_RUN", "independent_reference":"NOT_RUN",
                    "experimental_calibration":"NOT_RUN", "norm_check":
                    "PASS" if metrics.get("wave_intensity_conservation_within_0_1_percent") is True else "NOT_RUN"}
    summary = (str(metrics.get("illumination_scope",illumination)) + f" {modes} source/energy mode(s), {phonons} specimen configuration(s); "
               + str(metrics.get("specimen_model","specimen checkpoint")) + "; "
               + (str(old.get("aperture_policy",{}).get("strategy","physical recording planes")) if product=="TEM" else str(metrics.get("physical_detector_routing")))
               + "; backend " + str(metrics.get("wave_compute_backend",old.get("backend","not recorded"))))
    manifest = {**deepcopy(old), "schema":"execution-evidence-v2", "product":product,"nodes":nodes,"summary":summary,
                "source_energy_mode_count":modes,"phonon_configuration_count":phonons,
                "requested_backend":getattr(state,"acceleration_backend","Auto"),
                "requested_policy":getattr(state.sample,"stem_execution_policy","auto") if product=="STEM" else "auto",
                "actual_backend":metrics.get("wave_compute_backend",old.get("backend")),
                "fallback_reason":metrics.get("specimen_fallback_reason") or metrics.get("fft_fallback_reason"),
                "aberration_coverage":{"wave_terms":[term.name for term in WAVE_TERMS], "unsupported":list(UNIMPLEMENTED_TERMS),
                                       "effective_coefficients":deepcopy(metrics.get("effective_aberrations",{})),
                                       "scope":"declared residual phase basis; no inference that every term is field calibrated"},
                "electrical_controls":electrical,"geometry_admission":deepcopy(getattr(state,"_geometry_effects_diagnostics",{})),
                "verification":verification}
    metrics["wave_execution_manifest"] = thaw_json(freeze_json(manifest))
    metrics["image_formation_scope"] = summary
    return result
