"""Forward paraxial column transport in the captured static electric field.

The integrated momenta are dimensional canonical transverse momenta.  At
every RK stage the mechanical momentum magnitude follows K[eV]-phi[V],
including the residual gun field beyond its bookkeeping exit.  This is the
same paraxial column model as the magnetic fast path, not a full-angle Lorentz
model.  Thin optical actions remain explicit and conserve kinetic energy.
"""
from __future__ import annotations

import math
import numpy as np

from .compute_backend import BACKEND_CPU, BACKEND_NUMBA, BACKEND_CUDA, GPUExecutionError

Q = -1.602176634e-19
M = 9.1093837015e-31
C = 299792458.0
REST_EV = M*C*C/(-Q)

try:
    from numba import njit, prange, cuda
    from numba.extending import register_jitable
except ImportError:
    njit = cuda = None
    prange = range
    def register_jitable(function=None, **kwargs):
        return function if callable(function) else lambda wrapped: wrapped


def active_electric_field(plan):
    """Return the field unless its provider proves this whole span constant."""
    field = getattr(plan, "electric_field", None)
    constant = getattr(field, "is_constant_on_interval", None)
    if field is not None and callable(constant) and constant(float(plan.z_mm[0]), float(plan.z_mm[-1])):
        return None
    return field


@register_jitable
def _momentum_speed(energy_ev):
    ratio = energy_ev/REST_EV
    p = M*C*math.sqrt(ratio)*math.sqrt(ratio+2.)
    v = C*math.sqrt(ratio/(ratio+1.))*math.sqrt((ratio+2.)/(ratio+1.))
    return p, v


@register_jitable
def _cell_index(nodes, value):
    lower, upper = 0, len(nodes)
    while lower < upper:
        middle = (lower+upper)//2
        if value < nodes[middle]:
            upper = middle
        else:
            lower = middle+1
    index = lower-1
    if index < 0:
        index = 0
    if index > len(nodes)-2:
        index = len(nodes)-2
    return index


@register_jitable
def _grid_electric(x, y, z, data):
    r, zz, voltage = data
    radius = math.sqrt(x*x+y*y)
    if not math.isfinite(radius+z) or radius > r[-1] or z < zz[0] or z > zz[-1]:
        return math.nan, math.nan, math.nan, math.nan
    i, j = _cell_index(r,radius), _cell_index(zz,z)
    ds, dz = r[i+1]*r[i+1]-r[i]*r[i], zz[j+1]-zz[j]
    u, v = (radius*radius-r[i]*r[i])/ds, (z-zz[j])/dz
    a, b, c, d = voltage[i,j], voltage[i+1,j], voltage[i,j+1], voltage[i+1,j+1]
    lower, upper = a+u*(b-a), c+u*(d-c)
    radial = ((1.-v)*(b-a)+v*(d-c))/ds
    return lower+v*(upper-lower), -2.*x*radial, -2.*y*radial, -(upper-lower)/dz


@register_jitable
def _scalar_derivative(x, px, y, py, z, index, di, coefficients, forcing,
                       electric_data, invariant, optical_invariant, reference_p):
    phi, ex, ey, _ = _grid_electric(x, y, z, electric_data)
    energy, optical_energy = invariant+phi, optical_invariant+phi
    if not math.isfinite(energy+optical_energy) or energy <= 0. or optical_energy <= 0.:
        return math.nan, math.nan, math.nan, math.nan, math.nan
    p, speed = _momentum_speed(optical_energy)
    _, actual_speed = _momentum_speed(energy)
    b = coefficients[4,index]
    tx, ty = (px+b*y)/p, (py-b*x)/p
    kx, ky, hn, hs, kxy = (coefficients[0,index], coefficients[1,index],
                           coefficients[2,index], coefficients[3,index], coefficients[5,index])
    hu, hv = x*x-y*y, 2.*x*y
    dpx = -(reference_p*kx+b*b/p)*x-reference_p*kxy*y+(b/p)*py-reference_p*(hn*hu+hs*hv)
    dpy = -(reference_p*ky+b*b/p)*y-reference_p*kxy*x-(b/p)*px+reference_p*(hn*hv-hs*hu)
    dpx += -Q*forcing[1,di]+Q*ex/speed
    dpy += Q*forcing[0,di]+Q*ey/speed
    return tx, dpx, ty, dpy, math.sqrt(1.+tx*tx+ty*ty)/actual_speed


@register_jitable
def _scalar_step(x, tx, y, ty, z, z1, j, coefficients, forcing, electric_data,
                 invariant, optical_invariant, reference_p):
    h = z1-z
    phi = _grid_electric(x,y,z,electric_data)[0]
    energy = optical_invariant+phi
    if not math.isfinite(energy) or energy <= 0.:
        return math.nan, math.nan, math.nan, math.nan, math.nan
    p, _ = _momentum_speed(energy)
    a, b, c, di = 2*j, 2*j+1, 2*j+2, 3*j
    px, py = p*tx-coefficients[4,a]*y, p*ty+coefficients[4,a]*x
    k1 = _scalar_derivative(x,px,y,py,z,a,di,coefficients,forcing,electric_data,invariant,optical_invariant,reference_p)
    k2 = _scalar_derivative(x+.5*h*k1[0],px+.5*h*k1[1],y+.5*h*k1[2],py+.5*h*k1[3],z+.5*h,b,di+1,coefficients,forcing,electric_data,invariant,optical_invariant,reference_p)
    k3 = _scalar_derivative(x+.5*h*k2[0],px+.5*h*k2[1],y+.5*h*k2[2],py+.5*h*k2[3],z+.5*h,b,di+1,coefficients,forcing,electric_data,invariant,optical_invariant,reference_p)
    k4 = _scalar_derivative(x+h*k3[0],px+h*k3[1],y+h*k3[2],py+h*k3[3],z1,c,di+2,coefficients,forcing,electric_data,invariant,optical_invariant,reference_p)
    x += h*(k1[0]+2.*k2[0]+2.*k3[0]+k4[0])/6.
    px += h*(k1[1]+2.*k2[1]+2.*k3[1]+k4[1])/6.
    y += h*(k1[2]+2.*k2[2]+2.*k3[2]+k4[2])/6.
    py += h*(k1[3]+2.*k2[3]+2.*k3[3]+k4[3])/6.
    energy = optical_invariant+_grid_electric(x,y,z1,electric_data)[0]
    if not math.isfinite(energy) or energy <= 0.:
        return math.nan, math.nan, math.nan, math.nan, math.nan
    p, _ = _momentum_speed(energy)
    return x, (px+coefficients[4,c]*y)/p, y, (py-coefficients[4,c]*x)/p, h*(k1[4]+2.*k2[4]+2.*k3[4]+k4[4])/6.


@register_jitable
def _trace_one(ray, coordinates, initial_energy, optical_invariant, initial_time,
               z, coefficients, forcing, actions, electric_data, reference_p,
               save, checkpoint, history, checkpoints, error):
    x, tx, y, ty = coordinates[0,ray], coordinates[1,ray], coordinates[2,ray], coordinates[3,ray]
    invariant = initial_energy[ray]-_grid_electric(x,y,z[0],electric_data)[0]
    oi = invariant if math.isnan(optical_invariant) else optical_invariant
    time, saved, captured = initial_time[ray], 0, 0
    for j in range(len(z)):
        # These maps are defined as slope kicks, at unchanged kinetic energy.
        tx, ty = tx-actions[1,j]*x, ty-actions[1,j]*y
        if actions[2,j] != 0.:
            co, si = math.cos(actions[2,j]), math.sin(actions[2,j])
            x, y = co*x-si*y, si*x+co*y
            tx, ty = co*tx-si*ty, si*tx+co*ty
        tx, ty = tx+actions[3,j], ty+actions[4,j]
        radial = actions[0,j]*(x*x+y*y)
        tx, ty = tx-radial*x, ty-radial*y
        energy = invariant+_grid_electric(x,y,z[j],electric_data)[0]
        if not math.isfinite(x+tx+y+ty+energy) or energy <= 0.:
            error[ray] = j+1
        if saved < len(save) and j == save[saved]:
            history[0,saved,ray],history[1,saved,ray],history[2,saved,ray],history[3,saved,ray] = x,tx,y,ty
            history[4,saved,ray],history[5,saved,ray] = time,energy
            saved += 1
        if captured < len(checkpoint) and j == checkpoint[captured]:
            checkpoints[0,captured,ray],checkpoints[1,captured,ray],checkpoints[2,captured,ray],checkpoints[3,captured,ray] = x,tx,y,ty
            checkpoints[4,captured,ray],checkpoints[5,captured,ray] = time,energy
            captured += 1
        if j+1 < len(z):
            x,tx,y,ty,elapsed = _scalar_step(x,tx,y,ty,z[j],z[j+1],j,coefficients,forcing,electric_data,invariant,oi,reference_p)
            time += elapsed


def _parallel_impl(ray_start, ray_stop, coordinates, initial_energy, optical_invariant, initial_time,
                   z, coefficients, forcing, actions, electric_data, reference_p,
                   save, checkpoint, history, checkpoints, error):
    for ray in prange(ray_start, ray_stop):
        _trace_one(ray,coordinates,initial_energy,optical_invariant,initial_time,z,
                   coefficients,forcing,actions,electric_data,reference_p,save,checkpoint,history,checkpoints,error)


_parallel = njit(cache=True, nogil=True, parallel=True)(_parallel_impl) if njit is not None else None
_serial = njit(cache=True, nogil=True)(_parallel_impl) if njit is not None else None

if cuda is not None:
    @cuda.jit
    def _cuda_kernel(ray_start, ray_stop, coordinates, initial_energy, optical_invariant, initial_time,
                     z, coefficients, forcing, actions, electric_data, reference_p,
                     save, checkpoint, history, checkpoints, error):
        ray = cuda.grid(1)+ray_start
        if ray < ray_stop:
            _trace_one(ray,coordinates,initial_energy,optical_invariant,initial_time,z,
                       coefficients,forcing,actions,electric_data,reference_p,save,checkpoint,history,checkpoints,error)
else:
    _cuda_kernel = None


def _compiled_electric_data(field):
    from .closed_gun_field import ClosedGunField
    from .instrument_electric import InstrumentElectricField
    base = getattr(field, 'base_field', None)
    provider = getattr(field, 'provider', None)
    if (type(field) is not InstrumentElectricField
            or getattr(field.interpolate, '__func__', None) is not InstrumentElectricField.interpolate
            or type(base) is not ClosedGunField or getattr(provider, 'wien_field', None) is not None
            or getattr(base.interpolate, '__func__', None) is not ClosedGunField.interpolate):
        return None
    return tuple(np.ascontiguousarray(value) for value in (base.r, base.z, base.voltage))


def _vector_reference(inputs, z, electric, energy, optical_invariant, initial_time,
                      mapped_fields, step_operator, defer_nonfinite, cancel_check):
    (kx,ky,hn,hs,b,_,cs,power,rotation,step,x,tx,y,ty,kickx,kicky,save,cp,kxy,bx,by,ref) = inputs
    count = len(x)
    history = np.empty((6,len(save),count)); checkpoints = np.empty((6,len(cp),count))
    saved = captured = 0
    time = initial_time.copy()
    def sample(xx, yy, zz):
        points = np.column_stack((xx,yy,np.full(count,zz)))
        valid = np.all(np.isfinite(points),axis=1)
        bounds = getattr(electric, 'bounds_m', None)
        if bounds is not None:
            valid &= np.all((points >= bounds[0]) & (points <= bounds[1]),axis=1)
            valid &= np.hypot(points[:,0],points[:,1]) <= bounds[1,0]
        if not defer_nonfinite and not np.all(valid):
            raise ValueError('Electric column trajectory left the fixed instrument field domain')
        potential = np.full(count,np.nan); ef = np.full((count,3),np.nan)
        if np.any(valid):
            potential[valid],ef[valid] = electric.interpolate(points[valid])
        return potential,ef
    invariant = energy-sample(x,y,z[0])[0]
    oi = invariant if optical_invariant is None else optical_invariant
    def pv(values):
        ratios = values/REST_EV
        with np.errstate(invalid='ignore',divide='ignore'):
            momentum = M*C*np.sqrt(ratios)*np.sqrt(ratios+2.)
            speed = C*np.sqrt(ratios/(ratios+1.))*np.sqrt((ratios+2.)/(ratios+1.))
        return momentum,speed
    for j in range(len(z)):
        if cancel_check is not None and j % 64 == 0: cancel_check()
        tx,ty=tx-power[j]*x,ty-power[j]*y
        if rotation[j] != 0.:
            co,si=math.cos(rotation[j]),math.sin(rotation[j])
            x,y=co*x-si*y,si*x+co*y;tx,ty=co*tx-si*ty,si*tx+co*ty
        tx,ty=tx+kickx[j],ty+kicky[j]
        radial=cs[j]*(x*x+y*y);tx,ty=tx-radial*x,ty-radial*y
        phi,_=sample(x,y,z[j]); actual_energy=invariant+phi
        if saved<len(save) and j==save[saved]:
            history[:,saved]=x,tx,y,ty,time,actual_energy;saved+=1
        if captured<len(cp) and j==cp[captured]:
            checkpoints[:,captured]=x,tx,y,ty,time,actual_energy;captured+=1
        if j+1==len(z):break
        a,mid,c,d=2*j,2*j+1,2*j+2,3*j
        p,_=pv(oi+phi)
        before=tuple(v.copy() for v in (x,tx,y,ty))
        initial=np.array((x,p*tx-b[a]*y,y,p*ty+b[a]*x))
        active=tuple(item for item in mapped_fields if item.field_map.field_support_mm[0]<z[j+1]*1e3 and item.field_map.field_support_mm[1]>z[j]*1e3)
        def derivative(values,index,di,zz):
            xx,px,yy,py=values
            phi,ef=sample(xx,yy,zz);p,v=pv(oi+phi);_,va=pv(invariant+phi)
            ux,uy=(px+b[index]*yy)/p,(py-b[index]*xx)/p
            hu,hv=xx*xx-yy*yy,2.*xx*yy
            fx=-Q*by[di]+Q*ef[:,0]/v;fy=Q*bx[di]+Q*ef[:,1]/v
            if active:
                points=np.column_stack((xx,yy,np.full(count,zz)))
                good=np.all(np.isfinite(points),axis=1)
                magnetic=np.full((count,3),np.nan)
                if np.any(good):magnetic[good]=sum((item.field_at_global_positions_t(points[good]) for item in active),start=np.zeros((np.count_nonzero(good),3)))
                mx,my,mz=magnetic.T;factor=Q*np.sqrt(1.+ux*ux+uy*uy)
                fx+=factor*(uy*mz-(1.+ux*ux)*my+ux*uy*mx)
                fy+=factor*((1.+uy*uy)*mx-ux*mz-ux*uy*my)
            derivatives=np.array((ux,-(ref[0]*kx[index]+b[index]**2/p)*xx-ref[0]*kxy[index]*yy+(b[index]/p)*py-ref[0]*(hn[index]*hu+hs[index]*hv)+fx,
                                  uy,-(ref[0]*ky[index]+b[index]**2/p)*yy-ref[0]*kxy[index]*xx-(b[index]/p)*px+ref[0]*(hn[index]*hv-hs[index]*hu)+fy))
            return derivatives,np.sqrt(1.+ux*ux+uy*uy)/va
        h=step[j]
        ka,ta=derivative(initial,a,d,np.nextafter(z[j],z[j+1]))
        kb,tb=derivative(initial+.5*h*ka,mid,d+1,.5*(z[j]+z[j+1]))
        kc,tc=derivative(initial+.5*h*kb,mid,d+1,.5*(z[j]+z[j+1]))
        kd,td=derivative(initial+h*kc,c,d+2,np.nextafter(z[j+1],z[j]))
        x,px,y,py=initial+h*(ka+2.*kb+2.*kc+kd)/6.
        p,_=pv(oi+sample(x,y,z[j+1])[0]);tx,ty=(px+b[c]*y)/p,(py-b[c]*x)/p
        time+=h*(ta+2.*tb+2.*tc+td)/6.
        if step_operator is not None:
            step_operator.energies=invariant+sample(x,y,z[j+1])[0]
            x,tx,y,ty=step_operator(j,before,(x,tx,y,ty))
        if not defer_nonfinite and not np.all(np.isfinite((x,tx,y,ty))):
            raise ValueError('Electric column trajectory left its positive-energy forward paraxial domain')
    return history,checkpoints


def electrostatic_column_rk4(inputs, *, z_mm, electric_field, initial_kinetic_energy_ev,
                             optical_reference_invariant_ev=None, initial_time_s=None,
                             backend=BACKEND_CPU, policy='auto', mapped_fields=(),
                             step_operator=None, defer_nonfinite_until_clipping=False,
                             serial=False, cancel_check=None):
    """Return old ten transport arrays followed by saved/checkpoint energies."""
    count=len(inputs[10]);z=np.ascontiguousarray(z_mm)*1e-3
    energy=np.broadcast_to(np.asarray(initial_kinetic_energy_ev,float),(count,)).copy()
    finite_phase=np.all(np.isfinite(np.asarray(inputs[10:14])),axis=0)
    invalid_energy=~np.isfinite(energy)|(energy<=0.)
    if np.any(invalid_energy & (finite_phase | (not bool(defer_nonfinite_until_clipping)))):
        raise ValueError('Column entrance kinetic energies must be positive and finite')
    times=np.full(count,np.nan) if initial_time_s is None else np.array(initial_time_s,float,copy=True)
    data=_compiled_electric_data(electric_field)
    compiled=data is not None and not mapped_fields and step_operator is None
    reason=None
    if backend==BACKEND_CUDA and not compiled:
        if policy=='require_gpu':
            raise GPUExecutionError('unsupported_stage','This electric provider / mapped-field / medium column needs the reference CPU solver')
        backend=BACKEND_CPU;reason='electric provider, mapped field or collisions require the reference CPU solver'
    if backend==BACKEND_NUMBA and (not compiled or _parallel is None):
        backend=BACKEND_CPU;reason='electric provider, mapped field or collisions require the reference CPU solver'
    if backend in (BACKEND_NUMBA,BACKEND_CUDA):
        coeff=np.ascontiguousarray((inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[18]))
        forcing=np.ascontiguousarray((inputs[19],inputs[20]))
        actions=np.ascontiguousarray((inputs[6],inputs[7],inputs[8],inputs[14],inputs[15]))
        coords=np.ascontiguousarray(inputs[10:14]);save,cp=inputs[16:18]
        history=np.empty((6,len(save),count));checkpoints=np.empty((6,len(cp),count));error=np.zeros(count,np.int64)
        args=(coords,energy,math.nan if optical_reference_invariant_ev is None else float(optical_reference_invariant_ev),times,z,coeff,forcing,actions,data,float(inputs[21][0]),save,cp,history,checkpoints,error)
        if backend==BACKEND_CUDA:
            if _cuda_kernel is None:raise GPUExecutionError('unavailable','CUDA electric-column kernel is not available')
            device=[]
            for value in args:
                device.append(tuple(cuda.to_device(v) for v in value) if isinstance(value,tuple)
                              else cuda.to_device(value) if isinstance(value,np.ndarray) else value)
            for start in range(0,count,256):
                if cancel_check is not None: cancel_check()
                stop=min(start+256,count)
                _cuda_kernel[(stop-start+127)//128,128](start,stop,*device)
                cuda.synchronize()
            history,checkpoints,error=(device[i].copy_to_host() for i in (-3,-2,-1))
        else:
            for start in range(0,count,64):
                if cancel_check is not None: cancel_check()
                (_serial if serial else _parallel)(start,min(start+64,count),*args)
        if np.any(error) and not defer_nonfinite_until_clipping:
            raise ValueError('Electric column trajectory left its field domain or positive-energy forward paraxial domain')
    else:
        history,checkpoints=_vector_reference(inputs,z,electric_field,energy,optical_reference_invariant_ev,times,mapped_fields,step_operator,defer_nonfinite_until_clipping,cancel_check)
    outputs=(*history[:4],*checkpoints[:4],history[4],checkpoints[4],history[5],checkpoints[5])
    return outputs,backend,reason
