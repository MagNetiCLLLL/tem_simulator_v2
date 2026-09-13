"""Independent free-space check of the moving radial orbital frame.

This fixture is not an emitted source or image acceptance. Compare the
executed two-way operator against a direct Bessel angular-spectrum integral.
"""
import json
import math
import numpy as np
from scipy.special import roots_laguerre, j0
from threadpoolctl import threadpool_limits
from temsim.physics.radial_gun_wave import laguerre_operators, basis_values
from temsim.physics.scattering_load import hermitian_slab, outgoing_load
from temsim.physics.radial_coordinates import blended_radial_chart


def calculate(step, count=12, k=100., distance=100., transition=None, chart_override=None, quartic_chart=None):
    x, d, t = laguerre_operators(count)
    operators = []
    phase = (lambda z: (0., 0.)) if quartic_chart is None else quartic_chart
    if phase(0.)[0] != 0:
        raise ValueError("The reference free-wave fixture requires its unchanged initial phase")
    q = lambda z: 1j/k/(1+1j*z/k)
    target_width = 1. if transition is None else transition.target_width_over_cap_radius
    target_q = lambda z: (1j/(k*target_width**2))/(1+1j*z/(k*target_width**2))
    def chart(z):
        value, target = q(z), target_q(z)
        base = blended_radial_chart(value, -value*value, target, -target*target, k, z, transition)
        return base if chart_override is None else chart_override(z, base)
    for left in np.linspace(0., distance, round(distance/step)+1)[:-1]:
        width, curvature, rate, curvature_prime = chart(left+step/2)
        g = .5*k*curvature_prime*width**2*x+1j*rate*d
        kinetic = t/width**2+(k*curvature*width)**2*x-2j*k*curvature*d
        closure = np.zeros_like(kinetic)
        closure[-1, -1] = count**2*((.5*k*curvature_prime*width**2)**2+rate**2)
        quartic, prime = phase(left+step/2)
        if quartic or prime:
            from temsim.physics.quartic_radial_phase import quartic_operators
            kinetic, g, closure = quartic_operators(count, width, curvature, k, rate, curvature_prime, quartic, prime)
        op, _ = hermitian_slab(-kinetic-closure, g, step, k, carrier_k=k)
        operators.append(op)
    width, curvature, _, _ = chart(distance)
    kinetic = t/width**2+(k*curvature*width)**2*x-2j*k*curvature*d
    if phase(distance)[0]:
        from temsim.physics.quartic_radial_phase import quartic_operators
        kinetic, _, _ = quartic_operators(count, width, curvature, k, quartic=phase(distance)[0])
    load = outgoing_load(operators, k*k*np.eye(count)-kinetic, k)
    initial = np.zeros(count, complex); initial[0] = 1
    field, _ = load.propagate(initial)
    radius = np.linspace(0., 12., 2001)
    actual = basis_values(radius, width, curvature, k, count)@field[-1]*np.exp(-1j*k*distance)
    actual *= np.exp(1j*phase(distance)[0]*radius**4)
    nodes, weights = roots_laguerre(128)
    delta = -2*nodes/(np.sqrt((k*k-2*nodes).astype(complex))+k)
    exact = j0(radius[:, None]*np.sqrt(2*nodes))@(weights*np.exp(1j*delta*distance))/np.sqrt(np.pi)
    error = np.sqrt(np.trapezoid(2*np.pi*radius*abs(actual-exact)**2, radius))
    return {"step": step, "modes": count, "relative_complex_error": float(error)}


if __name__ == "__main__":
    with threadpool_limits(1):
        for step in (10., 5., 2.5, 1.25):
            print(json.dumps(calculate(step)), flush=True)
