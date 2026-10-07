"""Rigid placement of the existing post-gun transverse magnetic models.

Local A=(0,0,P(x,y) f(z)); P is the captured dipole/quadrupole/hexapole
polynomial and f is the original uniform or Gaussian axial envelope. Its
curl reproduces the existing transverse field exactly, without introducing
a new fringe or Maxwell multipole model. No gun or receiver pose is added.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from temsim.physics.lens_field_provider import CoordinateRegistration, _posed_axial_support


@dataclass(frozen=True)
class FrozenPosedMultipole:
    lens_key: str
    component_key: str
    registration: CoordinateRegistration
    polynomial_terms: tuple
    envelope_kind: str
    center_z_m: float
    sigma_m: float
    native_support_mm: tuple
    radial_support_m: float
    reference_momentum: float
    event_z_mm: float | None = None
    event_dx_rad: float = 0.
    event_dy_rad: float = 0.
    captured_time_s: float | None = None
    drive_keys: tuple = ()
    dynamic: bool = False

    def __post_init__(self):
        if self.envelope_kind not in {"gaussian", "uniform"}:
            raise ValueError("Posed multipole needs its original Gaussian or uniform envelope")
        terms = {}
        for x, y, value in self.polynomial_terms:
            if (int(x) != x or int(y) != y or min(x, y) < 0 or not 1 <= x+y <= 3
                    or not math.isfinite(value)):
                raise ValueError("Posed multipole requires finite degree-one to degree-three Az terms")
            terms[int(x), int(y)] = terms.get((int(x), int(y)), 0.)+float(value)
        support = tuple(map(float, self.native_support_mm))
        if (len(support) != 2 or not np.isfinite(support).all() or support[1] <= support[0]
                or not np.isfinite((self.center_z_m, self.sigma_m, self.radial_support_m, self.reference_momentum)).all()
                or min(self.sigma_m, self.radial_support_m, self.reference_momentum) <= 0.):
            raise ValueError("Posed multipole support, width, radius and reference momentum must be finite and positive")
        object.__setattr__(self, "polynomial_terms", tuple((x, y, v) for (x, y), v in sorted(terms.items())))
        object.__setattr__(self, "native_support_mm", support)
        object.__setattr__(self, "drive_keys", tuple(self.drive_keys))

    @property
    def key(self):
        return self.lens_key

    @property
    def scale(self):
        return float(any(value != 0. for _, _, value in self.polynomial_terms))

    @property
    def field_support_mm(self):
        return _posed_axial_support(self.native_support_mm, self.registration, self.radial_support_m)

    def native_field_support_mm(self):
        return self.native_support_mm

    @property
    def fingerprint(self):
        from temsim.immutable_json import json_digest
        return json_digest({"model": "posed-native-Az-multipole-v1", **asdict(self)})

    @property
    def bounds_m(self):
        radius = self.radial_support_m
        lo, hi = np.asarray(self.native_support_mm)*1e-3
        from itertools import product
        corners = np.array(tuple(product((-radius, radius), (-radius, radius), (lo, hi))))
        points = corners@self.registration.rotation_array.T+self.registration.origin_array_m
        return np.stack((points.min(axis=0), points.max(axis=0)))

    def _local(self, positions, xp):
        points = xp.asarray(positions, dtype=xp.float64)
        if points.shape[-1:] != (3,):
            raise ValueError("Posed multipole positions must end in XYZ")
        return (points-xp.asarray(self.registration.origin_array_m))@xp.asarray(self.registration.rotation_array)

    def _envelope(self, z, xp):
        low, high = np.asarray(self.native_support_mm)*1e-3
        inside = (z >= low) & (z <= high)
        if self.envelope_kind == "uniform":
            return inside.astype(xp.float64), xp.zeros_like(z), xp.zeros_like(z)
        u = (z-self.center_z_m)/self.sigma_m
        f = xp.where(inside, xp.exp(-.5*u*u), 0.)
        return f, -f*u/self.sigma_m, f*(u*u-1)/self.sigma_m**2

    def _polynomial(self, local, xp, dx=0, dy=0):
        x, y = local[..., 0], local[..., 1]
        result = xp.zeros_like(x)
        for px, py, value in self.polynomial_terms:
            if px >= dx and py >= dy:
                factor = (math.factorial(px)//math.factorial(px-dx))*(math.factorial(py)//math.factorial(py-dy))
                result += factor*value*x**(px-dx)*y**(py-dy)
        return result

    def vector_potential_at_global_positions_t_m(self, positions, xp=None):
        if xp is None:
            from temsim.physics.wave_device import array_module
            xp = array_module(positions)
        local = self._local(positions, xp)
        value = self._polynomial(local, xp)*self._envelope(local[..., 2], xp)[0]
        return value[..., None]*xp.asarray(self.registration.rotation_array[:, 2])

    def vector_potential_jet(self, position_m):
        """Global A/dA/ddA within a support side; edges are not differentiable."""
        local = self._local(position_m, np)
        f, df, ddf = self._envelope(local[..., 2], np)
        p, px, py = (self._polynomial(local, np, *orders) for orders in ((0, 0), (1, 0), (0, 1)))
        gradient = np.stack((px*f, py*f, p*df), axis=-1)
        hessian = np.empty((*local.shape[:-1], 3, 3))
        hessian[..., 0, 0] = self._polynomial(local, np, 2, 0)*f
        hessian[..., 1, 1] = self._polynomial(local, np, 0, 2)*f
        hessian[..., 0, 1] = hessian[..., 1, 0] = self._polynomial(local, np, 1, 1)*f
        hessian[..., 0, 2] = hessian[..., 2, 0] = px*df
        hessian[..., 1, 2] = hessian[..., 2, 1] = py*df
        hessian[..., 2, 2] = p*ddf
        rotation = self.registration.rotation_array
        direction = rotation[:, 2]
        global_gradient = gradient@rotation.T
        global_hessian = np.einsum("ai,...ij,bj->...ab", rotation, hessian, rotation)
        return ((p*f)[..., None]*direction,
                np.einsum("i,...j->...ij", direction, global_gradient),
                np.einsum("i,...jk->...ijk", direction, global_hessian))

    def field_at_global_positions_t(self, positions):
        local = self._local(positions, np)
        f = self._envelope(local[..., 2], np)[0]
        magnetic = np.stack((self._polynomial(local, np, 0, 1)*f,
                             -self._polynomial(local, np, 1, 0)*f, np.zeros_like(f)), axis=-1)
        return magnetic@self.registration.rotation_array.T

    def slope_derivative(self, positions_m, slopes, charge_over_p):
        positions, slopes = np.asarray(positions_m, float), np.asarray(slopes, float)
        if positions.shape != (len(slopes), 3) or slopes.shape != (len(slopes), 2) or not np.isfinite(slopes).all():
            raise ValueError("Posed multipole transport requires finite matching XYZ and slope arrays")
        direction = np.column_stack((slopes, np.ones(len(slopes))))@self.registration.rotation_array
        if np.any(direction[:, 2] <= 1e-10):
            raise ValueError("Posed multipole trajectory leaves the forward local-axis domain")
        local_slopes = direction[:, :2]/direction[:, 2, None]
        local = self._local(positions, np)
        f = self._envelope(local[:, 2], np)[0]
        bx = self._polynomial(local, np, 0, 1)*f
        by = -self._polynomial(local, np, 1, 0)*f
        factor = np.broadcast_to(np.asarray(charge_over_p), (len(slopes),))
        acceleration = np.column_stack((-factor*by, factor*bx, np.zeros(len(slopes))))
        acceleration = acceleration@self.registration.rotation_array.T
        direction = np.column_stack((local_slopes, np.ones(len(slopes))))@self.registration.rotation_array.T
        dz = direction[:, 2, None]
        return (acceleration[:, :2]*dz-direction[:, :2]*acceleration[:, 2, None])/dz**3

    def maximum_step_mm(self, requested_step_mm, momentum_kg_m_s):
        if not np.isfinite((requested_step_mm, momentum_kg_m_s)).all() or min(requested_step_mm, momentum_kg_m_s) <= 0:
            raise ValueError("Posed multipole step and momentum must be positive")
        radius = self.radial_support_m
        peak = sum(abs(value)*(px+py)*radius**(px+py-1) for px, py, value in self.polynomial_terms)
        scale_mm = self.sigma_m*1e3 if self.envelope_kind == "gaussian" else np.ptp(self.native_support_mm)
        return min(float(requested_step_mm), .1*scale_mm,
                   .02*momentum_kg_m_s/(1.602176634e-19*max(peak, 1e-30))*1e3)


def _radius(component):
    for name in ("mechanical_clear_bore_diameter_mm", "bore_diameter_mm"):
        value = float(getattr(component, name, 0.))
        if value > 0. and math.isfinite(value):
            return value*.0005
    value = float(getattr(component, "effective_aperture_radius_mm", 0.))
    if value > 0. and math.isfinite(value):
        return value*.001
    raise ValueError(f"{component.key}: posed magnetic field requires its physical clear bore")


def capture_posed_column_fields(state, *, include_hexapole=True, time_s=None, arrival_time=None, dipoles=None):
    """Freeze nonzero placements; zero-pose optics retain their original path.

    ``dipoles`` optionally restricts capture to events admitted by a particle
    plan. Dynamic wave callers freeze each coil at the arrival time of its
    placed global centre plane via ``arrival_time(z_mm)``; its original event
    plane remains the ownership key. No longitudinal pulse envelope is added.
    """
    from temsim.lens_pose import has_lens_pose, lens_pose_registration
    from temsim.physics.core import electron, FIELD_SIGMA_CUTOFF
    from temsim.physics.instrument_magnetic import column_dipole_fields
    from temsim.simulation_modes import is_ideal
    components = (*getattr(state, "stigmators", ()), *getattr(state, "corrector_elements", ()),
                  *getattr(state, "deflectors", ()))
    by_key = {str(component.key): component for component in components}
    posed_keys = {key for key, component in by_key.items()
                  if bool(getattr(component, "enabled", False)) and has_lens_pose(state, key)}
    if not posed_keys:
        return ()
    charge, momentum, _ = electron(state)
    ratio = momentum/charge
    result = []
    for key, component in by_key.items():
        if key not in posed_keys or not any(hasattr(component, name) for name in
                ("quadrupole_tensor_m2", "quadrupole_strength_m2", "hexapole_strength_components_m3")):
            continue
        from temsim.test_electron_compiled_laws import multipole_is_supported
        if not multipole_is_supported(component):
            raise ValueError(f"{key}: posed multipole requires its declared native Gaussian field model")
        z = float(component.z_mm)
        if hasattr(component, "quadrupole_tensor_m2"):
            kx, ky, kxy = map(float, component.quadrupole_tensor_m2(np.asarray(z)))
            if not math.isclose(kx, -ky, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(f"{key}: posed stigmator requires the existing trace-free normal/skew model")
            terms = ((2, 0, -.5*ratio*kx), (0, 2, .5*ratio*kx), (1, 1, -ratio*kxy))
            width = float(component.length_mm)/2.355
        elif hasattr(component, "quadrupole_strength_m2"):
            strength = float(component.quadrupole_strength_m2(np.asarray(z)))
            terms = ((2, 0, -.5*ratio*strength), (0, 2, .5*ratio*strength))
            width = float(component.effective_length_mm)/2.355
        elif hasattr(component, "hexapole_strength_components_m3"):
            if not include_hexapole or is_ideal(state):
                continue
            normal, skew = map(float, component.hexapole_strength_components_m3(np.asarray(z)))
            terms = ((3, 0, -ratio*normal/3), (1, 2, ratio*normal),
                     (2, 1, -ratio*skew), (0, 3, ratio*skew/3))
            width = float(component.effective_length_mm)/2.355
        else:
            continue
        support_method = getattr(component, "field_support_mm", None)
        support = (support_method() if callable(support_method) else
                   (z-FIELD_SIGMA_CUTOFF*width, z+FIELD_SIGMA_CUTOFF*width))
        result.append(FrozenPosedMultipole(key, key, lens_pose_registration(state, key), terms,
            "gaussian", z*.001, width*.001, support, _radius(component), float(momentum)))
    coils = column_dipole_fields(state, time_s=time_s) if dipoles is None else tuple(dipoles)
    for original in coils:
        key = original.key.rsplit(":", 1)[0]
        if key not in posed_keys:
            continue
        field = original
        if field.dynamic and arrival_time is not None:
            capture_time = float(arrival_time(field.arrival_z_mm))
            field = next(row for row in column_dipole_fields(state, time_s=capture_time) if row.key == original.key)
            if (field.lower_m, field.upper_m, field.event_z_mm) != (original.lower_m, original.upper_m, original.event_z_mm):
                raise ValueError("Time-dependent posed deflector changed its physical coil geometry")
        component = by_key[key]
        result.append(FrozenPosedMultipole(field.key, key, lens_pose_registration(state, key),
            ((1, 0, -field.by_t), (0, 1, field.bx_t)), "uniform", field.event_z_mm*.001,
            field.upper_m-field.lower_m, (field.lower_m*1e3, field.upper_m*1e3), _radius(component),
            field.reference_momentum, field.event_z_mm, field.event_dx_rad, field.event_dy_rad,
            field.captured_time_s, field.drive_keys, field.dynamic))
    return tuple(result)
