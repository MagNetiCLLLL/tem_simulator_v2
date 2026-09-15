"""Compiled evaluation of the existing cut-cell/regular-axis scalar field.

No fast-math, new field fit, source, cutoff or altered error tolerance. The
NumPy implementation in AxisRegularPotential remains the reference/fallback.
"""
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _evaluate(p, r, z, voltage, cut, lookup, origin, inverse, gradient, phi0, core_s, core_slope):
    potential = np.empty(len(p))
    electric = np.empty_like(p)
    for n in range(len(p)):
        radius = np.hypot(p[n,0],p[n,1])
        s = radius*radius
        i = min(np.searchsorted(r,radius,side="right")-1,len(r)-2)
        j = min(np.searchsorted(z,p[n,2],side="right")-1,len(z)-2)
        width = r[i+1]**2-r[i]**2
        dz = z[j+1]-z[j]
        u, v = (s-r[i]**2)/width, (p[n,2]-z[j])/dz
        a,b,c,d = voltage[i,j],voltage[i+1,j],voltage[i,j+1],voltage[i+1,j+1]
        value = (1-u)*(1-v)*a+u*(1-v)*b+(1-u)*v*c+u*v*d
        gs, gz = ((1-v)*(b-a)+v*(d-c))/width, ((1-u)*(c-a)+u*(d-b))/dz
        if cut[i,j]:
            value, gs, gz = 0.,0.,0.
            parent = 2*(i*(len(z)-1)+j)+int(v>u)
            for slot in range(2):
                index = lookup[parent,slot]
                if index < 0:
                    continue
                ds, zz = s-origin[index,0], p[n,2]-origin[index,1]
                b0 = inverse[index,0,0]*ds+inverse[index,0,1]*zz
                b1 = inverse[index,1,0]*ds+inverse[index,1,1]*zz
                if min(b0,b1) >= -1e-9 and b0+b1 <= 1+1e-9:
                    gs,gz = gradient[index,0],gradient[index,1]
                    value = phi0[index]+gs*ds+gz*zz
                    break
        elif s < core_s[j] or s < core_s[j+1]:
            bottom,top = (1-u)*a+u*b,(1-u)*c+u*d
            bs,ts = (b-a)/width,(d-c)/width
            if s < core_s[j]:
                bottom,bs = voltage[0,j]+s*core_slope[j],core_slope[j]
            if s < core_s[j+1]:
                top,ts = voltage[0,j+1]+s*core_slope[j+1],core_slope[j+1]
            value = (1-v)*bottom+v*top
            gs,gz = (1-v)*bs+v*ts,(top-bottom)/dz
        potential[n] = value
        electric[n,0],electric[n,1],electric[n,2] = -2*p[n,0]*gs,-2*p[n,1]*gs,-gz
    return potential,electric


compiled_evaluate = njit(cache=True)(_evaluate) if njit is not None else None
