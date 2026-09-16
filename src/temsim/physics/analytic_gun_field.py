"""Compiled evaluation of the existing compact polynomial gun potential.

Same extractor, electrostatic-lens and accelerator ramps and SI conversion as
the NumPy implementation. No new fit, field cutoff or changed physical law.
"""
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _evaluate(z, terms):
    result = np.zeros((4, len(z)))
    for j in range(len(z)):
        for start, end, amplitude in terms:
            length = max(end-start, 1e-9)
            u = (z[j]-start)/length
            t = min(1., max(0., u))
            result[0, j] += amplitude*t**3*(10.-15.*t+6.*t**2)
            if 0 < u < 1:
                result[1, j] += amplitude*(30.*t**2-60.*t**3+30.*t**4)/length
                result[2, j] += amplitude*(60.*t-180.*t**2+120.*t**3)/length**2
                result[3, j] += amplitude*(60.-360.*t+360.*t**2)/length**3
    return result


compiled_evaluate = njit(cache=True)(_evaluate) if njit is not None else None


def evaluate(field, z_mm):
    from temsim.optics.electron_gun.electrostatic import (
        ExtractorElectrode, ElectrostaticGunLens, AcceleratorColumn,
    )
    if compiled_evaluate is None or not getattr(field, "compiled_axial", True):
        return None
    extractor, lens, accelerator = field.extractor, field.electrostatic_lens, field.accelerator
    for obj, cls, method in ((extractor, ExtractorElectrode, "axial_potential_v_and_derivatives_per_mm"),
                            (lens, ElectrostaticGunLens, "axial_potential_v_and_derivatives_per_mm"),
                            (accelerator, AcceleratorColumn, "normalized_potential_and_derivatives_per_mm")):
        if type(obj) is not cls or method in obj.__dict__:
            return None  # Added or overridden fields use the general provider.
    start = extractor.transition_start_mm+extractor.field_center_offset_mm
    end = extractor.transition_end_mm+extractor.field_center_offset_mm
    terms = [(start, end, extractor.voltage_kv*1000.)]
    center = lens.optical_reference_from_tip_mm
    half, edge = .5*lens.mechanical_length_mm, max(lens.soft_edge_mm, 1e-6)
    amplitude = lens.voltage_kv*1000.*lens.potential_scale
    terms += [(center-half-edge, center-half+edge, amplitude),
              (center+half-edge, center+half+edge, -amplitude)]
    gain = accelerator.high_tension_kv*1000.-field.emitter.emission_energy_ev-extractor.voltage_kv*1000.
    previous = 0.
    for stage in accelerator.stages:
        center = stage.center_from_tip_mm+accelerator.field_center_offset_mm
        terms.append((center-stage.soft_edge_mm, center+stage.soft_edge_mm,
                      gain*(stage.voltage_fraction-previous)))
        previous = stage.voltage_fraction
    z = np.asarray(z_mm, dtype=float)
    values = compiled_evaluate(z.ravel(), np.asarray(terms, dtype=float))
    return tuple(row.reshape(z.shape) for row in values)
