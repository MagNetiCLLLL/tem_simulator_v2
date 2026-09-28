"""Exact packed magnetic laws shared by gun and diagnostic particle solvers.

Packing does not resample fields or discard weak tails. Unsupported fields
retain their native provider. Diagnostic transverse validity is an explicit
option; production uses the full captured sum and its own interception rules.
"""
import numpy as np
try:
    from numba import njit
except ImportError:
    njit = None


def prepare_compiled_magnetic_sources(source_records):
    """Return immutable numeric laws, or None if any active law is unsupported."""
    if njit is None:
        return None
    from temsim.test_electron_compiled_laws import lens_family, magnetic_provider_is_supported, multipole_is_supported
    from temsim.physics.lens_field_provider import GeometryAwareAnalyticFieldProvider
    from temsim.magnetic_field_scene import _MultipoleField
    from temsim.physics.instrument_magnetic import ColumnDipoleField
    from temsim.optics.lens_focal_length import raw_unit_field_peak
    from temsim.optics.electron_gun.alignment import GunDeflector, GunStigmator
    sources, terms = [], []
    for source in source_records:
        if source.known_zero:
            continue
        if source.field_map is not None:
            return None
        provider = source.provider
        if not magnetic_provider_is_supported(provider):
            return None
        start = len(terms)
        kind, derivative_step = 0., 0.
        if type(provider) is GeometryAwareAnalyticFieldProvider:
            native = provider.native_provider
            family = lens_family(native)
            if family is None:
                return None
            support = provider.field_support_mm()
            derivative_step = max(abs(support[1]-support[0]), 1.)*1e-6
            if family == 'objective':
                for prefix in ('upper', 'lower'):
                    scale = float(native.percent)/100.*float(native.polarity)*getattr(native, prefix+'_b0_t')
                    width, center = getattr(native, prefix+'_a_mm'), getattr(native, prefix+'_field_center_z_mm')
                    for term in getattr(native, prefix+'_gaussian'):
                        terms.append([0., center+term.offset*width, max(abs(term.sigma*width),1e-12), term.amplitude*scale, 0.,0.,0.,0.,0.])
            elif family in ('condenser', 'round', 'normalized_round'):
                lens = getattr(native, 'lens', native)
                scale = float(lens.polarity)*lens.scale()
                if bool(getattr(lens, 'normalise_profile_peak', False)):
                    scale /= max(raw_unit_field_peak(lens), 1e-15)
                for term in lens.gaussian:
                    terms.append([0., lens.z_mm+term.offset*lens.a_mm, max(abs(term.sigma*lens.a_mm),1e-12),term.amplitude,scale,0.,0.,0.,0.])
                for row in terms[start:]:
                    row[3] *= scale
            else:
                return None
        elif type(provider) is ColumnDipoleField:
            kind = 2.
            terms.append([0.,0.,0.,provider.bx_t,provider.by_t,0.,0.,0.,0.])
        elif type(provider) is _MultipoleField:
            from temsim.physics.core import multipole_focusing_fields, skew_quadrupole_field, hexapole_field_components
            kind = 1.
            components = (*provider.state.stigmators, *provider.state.corrector_elements)
            if len(components) != 1:
                return None
            component = components[0]
            if not multipole_is_supported(component):
                return None
            center = float(component.z_mm)
            width = float(getattr(component, 'effective_length_mm', getattr(component, 'length_mm', 0.)))/2.355
            if width <= 0.:
                return None
            sample = np.array([center])
            kx, ky = multipole_focusing_fields(sample, provider.state)
            kxy = skew_quadrupole_field(sample, provider.state)
            hn, hs = hexapole_field_components(sample, provider.state)
            terms.append([0.,center,width,float(kx[0]),float(ky[0]),float(kxy[0]),float(hn[0]),float(hs[0]),provider.momentum_over_charge])
        elif type(provider) is GunDeflector:
            kind = 3.
            coils = [(provider.upper_center_from_tip_mm, provider.upper_field_x_mt, provider.upper_field_y_mt),
                     (provider.lower_center_from_tip_mm, provider.lower_field_x_mt, provider.lower_field_y_mt)]
            if provider.beam_blanked:
                coils = [(provider.upper_center_from_tip_mm,0.,provider.blanking_field_y_mt)]
            for center,bx,by in coils:
                terms.append([0.,center+provider.field_center_offset_mm,.5*provider.coil_length_mm,bx*.001,by*.001,provider.soft_edge_mm,0.,0.,0.])
        elif type(provider) is GunStigmator:
            kind = 5.
            angle = np.deg2rad(provider.rotation_deg)
            terms.append([0.,provider.optical_reference_from_tip_mm,.5*provider.effective_length_mm,provider.gradient_t_per_m,np.cos(angle),provider.soft_edge_mm,np.sin(angle),0.,0.])
        else:
            return None
        sources.append([*source.bounds_m[0],*source.bounds_m[1],source.radius_m,kind,float(start),float(len(terms)),derivative_step])
    packed = (np.asarray(sources, float).reshape(-1, 11),
              np.asarray(terms, float).reshape(-1, 9))
    for array in packed:
        array.setflags(write=False)
    return packed


def _jit(function):
    return function if njit is None else njit(cache=True, nogil=True)(function)


@_jit
def _window(z,center,half,edge):
    edge = max(edge,1e-6)
    result = 0.
    for sign,face in ((1.,center-half),(-1.,center+half)):
        t = min(1.,max(0.,(z-(face-edge))/(2.*edge)))
        result += sign*t*t*t*(10.+t*(-15.+6.*t))
    return result


@_jit
def _axial(z,terms,start,stop):
    value = 0.
    for index in range(start,stop):
        row = terms[index]
        ratio = (z-row[1])/row[2]
        value += row[3]*np.exp(-.5*ratio*ratio)
    return value


@_jit
def compiled_magnetic_field(point, sources, terms, check_transverse=True):
    """Evaluate the same finite-support sum in the original source order."""
    b = np.zeros(3)
    x, y, z = point
    radius = np.hypot(x, y)
    for source in sources:
        if not source[2]<=z<=source[5]:
            continue
        if check_transverse and (radius>source[6] or not source[0]<=x<=source[3] or not source[1]<=y<=source[4]):
            return False,b
        kind,start,stop = int(source[7]),int(source[8]),int(source[9])
        zm = z*1000.
        if kind==0:
            dz = source[10]
            derivative = (_axial(zm+dz,terms,start,stop)-_axial(zm-dz,terms,start,stop))/(2.*dz*.001)
            b[0] -= .5*x*derivative
            b[1] -= .5*y*derivative
            b[2] += _axial(zm,terms,start,stop)
        elif kind==1:
            row = terms[start]
            envelope = np.exp(-.5*((zm-row[1])/row[2])**2)*row[8]
            u,v = x*x-y*y,2.*x*y
            b[0] += envelope*(-row[4]*y-row[5]*x+row[6]*v-row[7]*u)
            b[1] += envelope*(row[3]*x+row[5]*y+row[6]*u+row[7]*v)
        elif kind==2:
            b[0] += terms[start,3]
            b[1] += terms[start,4]
        elif kind==3:
            for k in range(start,stop):
                row = terms[k]
                envelope = _window(zm,row[1],row[2],row[5])
                b[0] += row[3]*envelope
                b[1] += row[4]*envelope
        elif kind==5:
            row = terms[start]
            envelope = _window(zm,row[1],row[2],row[5])
            cosine,sine = row[4],row[6]
            bu = row[3]*(-sine*x+cosine*y)*envelope
            bv = row[3]*(cosine*x+sine*y)*envelope
            b[0] += cosine*bu-sine*bv
            b[1] += sine*bu+cosine*bv
    return True, b


@_jit
def compiled_magnetic_batch(points, sources, terms):
    result = np.empty_like(points)
    for index in range(len(points)):
        _, result[index] = compiled_magnetic_field(points[index], sources, terms, False)
    return result
