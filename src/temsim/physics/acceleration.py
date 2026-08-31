"""Relativistic electron momentum after the active gun exit."""
import numpy as np
E=1.602176634e-19;M=9.1093837015e-31;C=299792458.0

def voltage_profile_kv(state,z_mm):
    z = np.asarray(z_mm, dtype=float)
    return np.full_like(z, float(state.beam_voltage_kv), dtype=float)
def momentum_profile(state,z_mm,energy_offset_ev=None):
    kinetic_ev=np.asarray(voltage_profile_kv(state,z_mm),float)*1000.0
    if energy_offset_ev is not None:
        kinetic_ev=kinetic_ev[...,None]+np.asarray(energy_offset_ev,float)
    kinetic=E*np.maximum(kinetic_ev,1e-9);rest=M*C*C
    return np.sqrt(kinetic*kinetic+2.0*kinetic*rest)/C
