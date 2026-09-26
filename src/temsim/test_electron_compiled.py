"""Optional compiled scalar E+B step for captured analytic particle scenes.

The electric grid and every active magnetic contribution are copied without
resampling. Unsupported providers retain the generic reference solver. Fastmath
and parallel arithmetic are deliberately disabled: this is the same local
field law and discrete-gradient step, not a lower-accuracy preview.
"""
from __future__ import annotations
from math import sqrt
from dataclasses import replace
import numpy as np
from temsim.physics.relativistic_lorentz import ELEMENTARY_CHARGE_C as Q, ELECTRON_MASS_KG as M, SPEED_OF_LIGHT_M_PER_S as C
from temsim.test_electron_compiled_laws import lens_family, magnetic_provider_is_supported, multipole_is_supported
try:
    from numba import njit
except ImportError:
    njit = None


def prepare_compiled_fields(scene):
    """Return immutable packed input, or None for the retained general path."""
    if njit is None:
        return None
    from temsim.test_electron_scene import TestElectronScene
    from temsim.physics.closed_gun_field import ClosedGunField
    from temsim.physics.lens_field_provider import GeometryAwareAnalyticFieldProvider
    from temsim.magnetic_field_scene import _EquivalentDeflectorField, _MultipoleField
    from temsim.optics.lens_focal_length import unit_field_peak
    from temsim.optics.electron_gun.alignment import GunDeflector, GunStigmator
    if type(scene) is not TestElectronScene or type(scene.electric_base) is not ClosedGunField:
        return None
    for name in ('diagnostic_fields_at_global_position', 'diagnostic_position_is_valid', 'diagnostic_spatial_step_m'):
        if getattr(getattr(scene, name), '__func__', None) is not getattr(TestElectronScene, name):
            return None
    if getattr(scene.electric_base.interpolate, '__func__', None) is not ClosedGunField.interpolate:
        return None
    if getattr(scene.electric_provider, 'wien_field', None) is not None:
        return None
    sources, terms = [], []
    for source in scene.magnetic_scene._sources:
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
                    if family == 'condenser':
                        scale /= max(unit_field_peak(replace(lens, normalise_profile_peak=False)), 1e-15)
                    elif family == 'normalized_round':
                        # This provider normalizes each scalar evaluation by its
                        # own magnitude; represent that existing law explicitly.
                        kind = 4.
                    else:
                        return None
                for term in lens.gaussian:
                    terms.append([0., lens.z_mm+term.offset*lens.a_mm, max(abs(term.sigma*lens.a_mm),1e-12),term.amplitude,scale,0.,0.,0.,0.])
                if kind != 4.:
                    for row in terms[start:]:
                        row[3] *= scale
            else:
                return None
        elif type(provider) is _EquivalentDeflectorField:
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
    base = scene.electric_base
    packed = (np.ascontiguousarray(base.r),np.ascontiguousarray(base.z),np.ascontiguousarray(base.voltage),
              np.ascontiguousarray(scene.diagnostic_bounds_m),np.asarray(sources,float).reshape(-1,11),np.asarray(terms,float).reshape(-1,9))
    for array in packed:
        array.setflags(write=False)
    return packed


def _jit(function):
    return function if njit is None else njit(cache=True, nogil=True)(function)


@_jit
def _norm(v):
    return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])


@_jit
def _cross(a,b):
    return np.array((a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]))


@_jit
def _window(z,center,half,edge):
    edge = max(edge,1e-6)
    result = 0.
    for sign,face in ((1.,center-half),(-1.,center+half)):
        t = min(1.,max(0.,(z-(face-edge))/(2.*edge)))
        result += sign*t*t*t*(10.+t*(-15.+6.*t))
    return result


@_jit
def _axial(z,terms,start,stop,normalized):
    value = 0.
    for index in range(start,stop):
        row = terms[index]
        ratio = (z-row[1])/row[2]
        value += row[3]*np.exp(-.5*ratio*ratio)
    if normalized:
        value = value/max(abs(value),1e-15)*terms[start,4]
    return value


@_jit
def compiled_fields(point,data):
    r,zz,voltage,bounds,sources,terms = data
    b,e = np.zeros(3),np.zeros(3)
    x,y,z = point
    radius = np.hypot(x,y)
    for axis in range(3):
        if point[axis]<bounds[0,axis] or point[axis]>bounds[1,axis]:
            return False,b,e,0.
    if radius>r[-1] or z>zz[-1]:
        return False,b,e,0.
    potential = 0.
    if z>=0.:
        i = min(max(np.searchsorted(r,radius,side='right')-1,0),len(r)-2)
        j = min(max(np.searchsorted(zz,z,side='right')-1,0),len(zz)-2)
        low = r[i]*r[i]
        ds,dz = r[i+1]*r[i+1]-low,zz[j+1]-zz[j]
        u,v = (radius*radius-low)/ds,(z-zz[j])/dz
        a,bb,c,d = voltage[i,j],voltage[i+1,j],voltage[i,j+1],voltage[i+1,j+1]
        bottom,top = a+u*(bb-a),c+u*(d-c)
        potential = bottom+v*(top-bottom)
        ds_phi = ((1.-v)*(bb-a)+v*(d-c))/ds
        e[0],e[1],e[2] = -2.*x*ds_phi,-2.*y*ds_phi,-(top-bottom)/dz
    for source in sources:
        if not source[2]<=z<=source[5]:
            continue
        if radius>source[6] or not source[0]<=x<=source[3] or not source[1]<=y<=source[4]:
            return False,b,e,0.
        kind,start,stop = int(source[7]),int(source[8]),int(source[9])
        zm = z*1000.
        if kind==0 or kind==4:
            dz = source[10]
            derivative = (_axial(zm+dz,terms,start,stop,kind==4)-_axial(zm-dz,terms,start,stop,kind==4))/(2.*dz*.001)
            b[0] -= .5*x*derivative
            b[1] -= .5*y*derivative
            b[2] += _axial(zm,terms,start,stop,kind==4)
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
    return True,b,e,potential


@_jit
def _collinear(v,e,b):
    norm = _norm(v)
    axis = v/norm if norm>0. else e
    axisnorm = _norm(axis)
    if axisnorm==0.:
        return True
    return _norm(_cross(axis,e))<=1e-14*axisnorm*_norm(e) and _norm(_cross(axis,b))<=1e-14*axisnorm*_norm(b)


@_jit
def compiled_step(data,x0,u0,e0,phi0,dt):
    """Same fixed point and stopping tolerance; status 0=good, 1=outside, 2=iteration failure."""
    alpha = -Q*dt/(M*C)
    gamma0 = sqrt(1.+np.dot(u0,u0))
    u1 = u0+alpha*e0
    empty = np.zeros(3)
    for _ in range(32):
        gamma1 = sqrt(1.+np.dot(u1,u1))
        vbar = C*(u1+u0)/(gamma1+gamma0)
        dx = dt*vbar
        x1 = x0+dx
        valid,b,e,phi = compiled_fields(.5*(x0+x1),data)
        valid1,b1,e1,phi1 = compiled_fields(x1,data)
        if not valid or not valid1:
            return 1,x1,u1,b1,e1,phi1,0.
        if not _collinear(vbar,e,b) and Q*dt*_norm(b)/(M*.5*(gamma1+gamma0))>.05*(1.+1e-12):
            return 2,x1,u1,b1,e1,phi1,0.
        length2 = np.dot(dx,dx)
        potential_difference = phi1-phi0
        midpoint_work = np.dot(e,dx)
        resolution = 64.*np.finfo(np.float64).eps*max(abs(phi1),abs(phi0),1.)
        if length2>0. and abs(potential_difference)+abs(midpoint_work)>resolution:
            e = e-((potential_difference+midpoint_work)/length2)*dx
        t = alpha*C*b/(gamma1+gamma0)
        a = 2.*u0+alpha*e
        updated = (a+_cross(a,t)+t*np.dot(a,t))/(1.+np.dot(t,t))-u0
        if not np.isfinite(updated).all():
            return 2,x1,u1,b1,e1,phi1,0.
        scale = max(_norm(u0),_norm(updated),1e-9)
        if _norm(updated-u1)<=2e-11*scale:
            midpoint = .5*(u0+updated)
            speed0 = _norm(C*u0/sqrt(1.+np.dot(u0,u0)))
            speed1 = _norm(C*updated/sqrt(1.+np.dot(updated,updated)))
            speedmid = _norm(C*midpoint/sqrt(1.+np.dot(midpoint,midpoint)))
            path = dt*(speed0+4.*speedmid+speed1)/6.
            return 0,x1,updated,b1,e1,phi1,path
        u1 = updated
    return 2,x0,u1,empty,empty,0.,0.

@_jit
def _spatial_step(point,direction,requested,data,sampling):
    r,zz,_,_,_,_ = data
    regions,edges,sampling_bounds,sampling_steps = sampling
    z,nz = point[2],direction[2]
    step = requested
    if abs(nz)>1e-14:
        for edge in edges:
            ahead = (edge-z)/nz
            if ahead>max(1e-13,requested*1e-10):
                step = min(step,ahead)
        for box in regions:
            if box[0,2]<=z<=box[1,2]:
                step = min(step,(box[1,2]-box[0,2])/(96.*abs(nz)))
    transverse = sqrt(direction[0]**2+direction[1]**2)
    for i in range(len(sampling_bounds)):
        if sampling_bounds[i,0,2]<=z<=sampling_bounds[i,1,2]:
            step = min(step,sampling_steps[i,0])
            if transverse>1e-14:
                step = min(step,sampling_steps[i,1]/transverse)
    if z<=zz[-1]:
        radius = np.hypot(point[0],point[1])
        ir = min(max(np.searchsorted(r,radius,side='right')-1,0),len(r)-2)
        iz = min(max(np.searchsorted(zz,z,side='right')-1,0),len(zz)-2)
        transverse = np.hypot(direction[0],direction[1])
        if transverse>1e-14:
            step = min(step,.5*(r[ir+1]-r[ir])/transverse)
            for face,neighbour in ((r[ir],ir-1),(r[ir+1],ir+1)):
                if 0<=neighbour<len(r)-1:
                    into_next = .5*(r[neighbour+1]-r[neighbour])
                    step = min(step,(abs(radius-face)+into_next)/transverse)
        if abs(nz)>1e-14:
            following = min(len(zz)-2,iz+1) if nz>0. else max(0,iz-1)
            step = min(step,.5*(zz[iz+1]-zz[iz])/abs(nz))
            face = zz[iz+1] if nz>0. else zz[iz]
            into_next = .5*(zz[following+1]-zz[following])
            step = min(step,(abs(face-z)+into_next)/abs(nz))
    return step


@_jit
def _curved_step(point,velocity,curvature,requested,edges):
    if curvature<=0. or len(edges)==0:
        return requested
    z,nz = point[2],velocity[2]
    epsilon = max(2e-15,np.spacing(abs(z))*16.)
    step = requested
    for sign in (1.,-1.):
        distance = np.inf
        for edge in edges:
            offset = sign*(edge-z)
            if offset>epsilon:
                distance = min(distance,offset)
        if not np.isfinite(distance):
            continue
        speed = sign*nz
        discriminant = sqrt(speed*speed+2.*curvature*distance)
        limit = 2.*distance/(speed+discriminant) if speed>=0. else (discriminant-speed)/curvature
        step = min(step,limit)
    return step

# Imported after the scalar field kernels so the optional interception module
# can share the same optional-JIT policy without circular numerical calls.
from temsim.test_electron_intercepts import compiled_intercept


@_jit
def compiled_block(data,sampling,intercepts,unsupported,x,u,b,e,phi,elapsed,distance,suggested_dt,
                   max_path,step,relative_tol,position_tol,u_floor,block_steps):
    """Execute a bounded block of accepted states; Python owns progress/cancellation."""
    rows = np.empty((block_steps,9))
    count,status = 0,0
    path_tolerance = max(max_path*2e-12,position_tol*.01)
    for _ in range(block_steps):
        remaining = max_path-distance
        if remaining<=path_tolerance:
            status = 1
            break
        velocity = C*u/sqrt(1.+np.dot(u,u))
        speed = _norm(velocity)
        direction = velocity/speed if speed>0. else np.zeros(3)
        requested = _spatial_step(x,direction,min(step,remaining),data,sampling)
        electric_norm,magnetic_norm = _norm(e),_norm(b)
        rotation_field = 0. if _collinear(velocity,e,b) else magnetic_norm
        acceleration_bound = Q/M*(electric_norm+C*rotation_field)
        if acceleration_bound>0.:
            dt = 2.*requested/(speed+sqrt(speed*speed+2.*acceleration_bound*requested))
        elif speed>0.:
            dt = requested/speed
        else:
            status = 5
            break
        if electric_norm>0.:
            dt = min(dt,.05*max(_norm(u),u_floor)*M*C/(Q*electric_norm))
        if rotation_field>0.:
            dt = min(dt,.05*sqrt(1.+np.dot(u,u))*M/(Q*magnetic_norm))
        dt = _curved_step(x,velocity,acceleration_bound,min(dt,suggested_dt),sampling[1])
        accepted = False
        pending_stop,last_failure = -1,5
        for attempt in range(64):
            pending_stop = -1
            if dt<=max(np.spacing(elapsed)*8.,1e-30):
                status = last_failure
                break
            full = compiled_step(data,x,u,e,phi,dt)
            first = compiled_step(data,x,u,e,phi,.5*dt) if full[0]==0 else full
            second = compiled_step(data,first[1],first[2],first[4],first[5],.5*dt) if first[0]==0 else first
            failed = full if full[0]!=0 else first if first[0]!=0 else second
            if failed[0]!=0:
                last_failure = 5
                if failed[0]==1:
                    last_failure = 4
                    start = first[1] if full[0]==0 and first[0]==0 else x
                    fraction,index = compiled_intercept(start,failed[1],intercepts)
                    if index>=0 and unsupported[index]:
                        last_failure = 100+index
                dt *= .5
                continue
            path = first[6]+second[6]
            position_scale = position_tol+relative_tol*max(path,full[6])
            momentum_scale = relative_tol*max(_norm(u),_norm(second[2]),u_floor*.01)
            error = max(_norm(second[1]-full[1])/position_scale,_norm(second[2]-full[2])/momentum_scale,
                        abs(path-full[6])/position_scale)
            if error>1.:
                dt *= max(.1,.8*error**(-1./3.))
                continue
            if path>remaining+path_tolerance:
                dt *= max(.1,remaining/path)
                continue
            fraction,index = compiled_intercept(x,first[1],intercepts)
            if index>=0:
                fraction *= .5
            else:
                fraction,index = compiled_intercept(first[1],second[1],intercepts)
                fraction = .5+.5*fraction
            if index>=0:
                pending_stop = index
                if fraction<=1e-12:
                    status = 100+index
                    break
                if (1.-fraction)*path>position_tol:
                    dt *= fraction
                    continue
            accepted = True
            suggested_dt = np.inf if error<1e-6 else dt*min(2.,max(.5,.9*error**(-1./3.)))
            break
        if not accepted:
            if status==0:
                status = last_failure
            break
        x,u,b,e,phi = second[1:6]
        elapsed += dt
        distance += first[6]+second[6]
        rows[count,:3],rows[count,3:6] = x,u
        rows[count,6],rows[count,7],rows[count,8] = elapsed,distance,phi
        count += 1
        if pending_stop>=0:
            status = 100+pending_stop
            break
    if status==0 and max_path-distance<=path_tolerance:
        status = 1
    return rows[:count],b,e,phi,suggested_dt,status
