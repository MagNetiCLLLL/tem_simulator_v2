import math
E=1.602176634e-19;M=9.1093837015e-31;C=299792458.;H=6.62607015e-34
def electron_properties(kv):
    k=E*kv*1000.;r=M*C*C;p=math.sqrt(k*k+2*k*r)/C
    return {'charge_c':E,'momentum':p,'wavelength_m':H/p}
