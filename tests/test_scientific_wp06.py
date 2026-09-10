"""Fixed scientific cases. Each property states its scope and measured error."""
import json
import numpy as np
import pytest
from types import SimpleNamespace
from temsim.physics.axisymmetric_magnetostatics import solve_axisymmetric
from temsim.physics.reference_benchmarks import finite_solenoid_axis, independent_flux_fvm


def _axes(n, extent=.06):
    r = np.unique(np.r_[np.linspace(0, .012, n), np.geomspace(.012, extent, 30)])
    z = np.unique(np.r_[-np.geomspace(.012, extent, 30), np.linspace(-.012, .012, 2*n-1), np.geomspace(.012, extent, 30)])
    return r, z


def _current(r,z):
    return ((r>=.004)&(r<=.006)&(np.abs(z)<=.005))*5e6


def test_finite_solenoid_fem_against_biot_savart(record_property):
    results = []
    points = np.linspace(-.01,.01,81)
    analytic = finite_solenoid_axis(points)
    for n in (49,97,145):
        r,z = _axes(n)
        solution = solve_axisymmetric(r,z, lambda r,z: np.ones_like(r), _current)
        values = np.interp(points,z,solution.bz_t[0])
        results.append({"near_radial_nodes":n,"axis_field_t":values.tolist(), "max_error_t":float(np.max(np.abs(values-analytic))),
                        "relative_l2":float(np.linalg.norm(values-analytic)/np.linalg.norm(analytic))})
    # 1% field accuracy is scoped to this finite winding and remote domain.
    assert results[-1]["relative_l2"] < .01
    assert results[-1]["max_error_t"] < 1e-4
    record_property("wp06_solenoid", json.dumps({"case":"100 A-turn annular finite solenoid", "reference":"Biot-Savart axial integral", "units":"T, m, A-turn", "boundary":"FEM A=0 at 60 mm; analytic unbounded vacuum", "z_m":points.tolist(),"reference_t":analytic.tolist(),"results":results}))


def test_simplified_pole_fem_against_independent_flux_fvm(record_property):
    # Two annular constant-mu pole bodies with a central gap; their material is
    # synthetic mu_r=50, not an OEM B-H calibration.
    def permeability(r,z):
        return np.where((r>=.002)&(r<=.009)&(np.abs(z)>=.001)&(np.abs(z)<=.004),50.,1.)
    points = np.linspace(-.00075,.00075,21)
    rows=[]
    for n in (49,97,145):
        r,z=_axes(n)
        fem=solve_axisymmetric(r,z,permeability,_current)
        fvm=independent_flux_fvm(r,z,permeability,_current)
        a,b=np.interp(points,z,fem.bz_t[0]),np.interp(points,z,fvm)
        rows.append({"near_radial_nodes":n,"fem_t":a.tolist(),"fvm_t":b.tolist(),"relative_l2":float(np.linalg.norm(a-b)/np.linalg.norm(b)),"max_error_t":float(np.max(np.abs(a-b)))})
    assert rows[-1]["relative_l2"] < .03
    assert rows[-1]["max_error_t"] < .002
    record_property("wp06_poles",json.dumps({"reference":"independent flux finite volumes vs A-phi triangles", "material":"synthetic constant mu_r=50", "boundary":"same 60 mm zero-potential domain", "scope":"implementation cross-check; no experimental calibration", "z_m":points.tolist(),"results":rows}))


def test_si_multislice_refinement_and_high_angle_coverage(record_property):
    from test_stem_cuda_pipeline import _state, _incident_bundle, _scan
    from temsim.physics.stem_wave_imaging import AngularDetector, simulate_angle_resolved_stem
    from temsim.physics.illumination import default_illumination_config
    detectors=(AngularDetector("bf",0,10),AngularDetector("df",30,45),AngularDetector("haadf",60,80))
    cases={"grid": [(128,8.,1.,2/3,4),(192,8.,1.,2/3,4),(256,8.,1.,2/3,4)],
           "fov": [(128,8.,1.,2/3,4),(192,12.,1.,2/3,4),(256,16.,1.,2/3,4)],
           "slice": [(192,8.,4.,2/3,4),(192,8.,2.,2/3,4),(192,8.,1.,2/3,4)],
           "bandwidth": [(192,8.,1.,.6,4),(192,8.,1.,2/3,4),(192,8.,1.,.8,4)],
           "phonons": [(192,8.,1.,2/3,2),(192,8.,1.,2/3,4),(192,8.,1.,2/3,8)]}
    cache, studies={},{}
    for name, configurations in cases.items():
        rows=[]
        for parameters in configurations:
            if parameters not in cache:
                n,fov,slice_step,bw,phonons=parameters
                state=_state("CPU",atomistic=True)
                state.sample.wave_grid_pixels=n
                state.sample.wave_field_of_view_angstrom=fov
                state.sample.wave_slice_thickness_angstrom=slice_step
                state.sample.wave_bandwidth_fraction=bw
                state.sample.wave_frozen_phonon_configurations=phonons
                state.sample.wave_illumination=default_illumination_config()
                state.sample.wave_illumination["pupil"]["semi_axes_mrad"]=[10.,10.]
                result=simulate_angle_resolved_stem(state,SimpleNamespace(incident=_incident_bundle()),detectors,*_scan())
                assert result.metrics["angular_coverage_complete"]
                cache[parameters]={"requested_grid":n,"fov_angstrom":fov,"slice_angstrom":slice_step,"bandwidth":bw,"phonons":phonons,
                    "maximum_angle_mrad":result.maximum_isotropic_angle_mrad,
                    "fractions":{k:float(np.mean(v)) for k,v in result.fractions.items()},
                    "sem":result.metrics["detector_configuration_relative_standard_error"],
                    "norm_drift":result.metrics["specimen_maximum_relative_intensity_change"]}
            rows.append(cache[parameters])
        # Report statistical and discretization convergence separately. Absolute
        # 5e-4 probability floor avoids meaningless relative error on tiny tails.
        deltas={k:abs(rows[-1]["fractions"][k]-rows[-2]["fractions"][k]) for k in ("bf","df","haadf")}
        for k,delta in deltas.items():
            stochastic=3*sum(row["sem"][k]*row["fractions"][k] for row in rows[-2:])
            assert delta <= 5e-4 + .05*abs(rows[-1]["fractions"][k]) + stochastic, (name,k,delta,rows)
        studies[name]={"cases":rows,"last_delta":deltas}
    record_property("wp06_multislice",json.dumps({"case":"Si [110], 0.4 nm thick, 10 mrad explicit pupil, 2x2 scan", "scope":"thin specimen local convergence; not a 5 nm production accuracy claim", "probability_floor":5e-4,"relative_tolerance":.05,"stochastic_tolerance":"3 times sum of independent SEM estimates", "studies":studies}))


def test_external_abtem_split_propagation_on_identical_potential(record_property):
    import abtem
    from abtem.multislice import FresnelPropagator
    from abtem.potentials.iam import PotentialArray
    from abtem.core.energy import energy2wavelength
    from temsim.physics.multislice import propagate_multislice
    from temsim.physics.wave_imaging import interaction_constant_rad_per_v_angstrom
    n=64; spacing=.125; energy=200_000.
    axis=(np.arange(n)-n//2)*spacing
    x,y=np.meshgrid(axis,axis,indexing="xy")
    wave=np.exp(-(x*x+y*y)/2+1j*.1*x); wave/=np.linalg.norm(wave)
    slices=np.array([4*np.exp(-((x-.1*i)**2+y*y)/.3) for i in range(4)])
    dz=np.array([.5,.5,.5,.5])
    ours,diagnostics=propagate_multislice(wave,slices,pixel_size_angstrom=spacing,
        wavelength_angstrom=energy2wavelength(energy),interaction_constant_rad_per_v_angstrom=interaction_constant_rad_per_v_angstrom(200),
        total_thickness_angstrom=None,target_slice_thickness_angstrom=.5,slice_thicknesses_angstrom=dz,bandwidth_fraction=2/3,compute_backend="NumPy CPU")
    other=abtem.Waves(wave.T.copy(),energy=energy,sampling=spacing)
    propagator=FresnelPropagator()
    # Both solvers use symmetric half propagation and the same hard circular
    # bandwidth. abTEM's usual transmission bandlimit is deliberately excluded
    # by using its transmission object directly; this tests propagation only.
    with abtem.config.set({"antialias.cutoff":2/3,"antialias.taper":0.}):
        for potential,thickness in zip(slices,dz):
            other=propagator.propagate(other,thickness/2,in_place=False)
            transmission=PotentialArray(potential.T[None],slice_thickness=thickness,sampling=spacing).transmission_function(energy)
            other=transmission.transmit(other)
            other=propagator.propagate(other,thickness/2,in_place=False)
    reference=np.asarray(other.array).squeeze().T
    error=float(np.linalg.norm(ours-reference)/np.linalg.norm(reference))
    assert error < 3e-6
    record_property("wp06_abtem_propagation",json.dumps({"version":abtem.__version__,"relative_complex_l2":error,"tolerance":3e-6,
        "independent":"abTEM transmission, FFT and Fresnel propagator", "shared":"identical supplied synthetic potential, boundary and bandwidth; does not validate potential construction", "dose":"one electron incident; no survival renormalization"}))


def test_at33_fixed_control_field_crystal_aperture_camera_chain(record_property):
    from test_interactive_calculation import small_real_state
    from temsim.simulation_pipeline import calculate
    from temsim.calculation_manifest import solver_source_identity
    state=small_real_state()
    state.illumination_mode="TEM"; state.projector_mode="image"
    for detector in state.stem_detectors: detector.inserted=False
    state.fluorescent_screen.inserted=False; state.camera.inserted=True
    state.sample.specimen_mode="reference"; state.sample.reference_sample_key="si_110"
    state.sample.wave_enabled=True; state.sample.wave_grid_pixels=64
    state.sample.wave_field_of_view_angstrom=16.; state.sample.thickness_nm=.4
    state.sample.wave_multislice_enabled=True; state.sample.wave_atomistic_enabled=True
    state.sample.wave_frozen_phonon_enabled=False
    state.objective_aperture.enabled=True; state.objective_aperture.radius_mm=.03
    state.simulation_mode="custom"
    state.lens_field_map_descriptors[state.objective_lens.key]={"solver":"axisymmetric_linear_fem","ampere_turns":250,
        "relative_permeability":100,"radial_nodes":24,"axial_nodes":48,"padding_factor":2.,"geometry_policy":"authoritative_dimensions"}
    before=[(l.key,l.percent,l.polarity) for l in state.lenses]
    result=calculate(state)
    assert result.wave_imaging is not None
    assert before==[(l.key,l.percent,l.polarity) for l in state.lenses]
    metrics=result.wave_imaging.metrics
    assert np.all(np.isfinite(result.wave_imaging.camera_intensity))
    assert metrics["specimen_atomistic_applied"]
    assert any(r["physical_element_id"]=="aperture:"+state.objective_aperture.key for r in metrics["camera_flux_ledger"])
    assert 0 < metrics["camera_collected_zero_loss_relative_intensity"] <= 1+1e-6
    record_property("wp06_full_chain",json.dumps({"solver_sha256":solver_source_identity(),"controls":before,
        "field_recipe":state.lens_field_map_descriptors,"sample":"Si [110], 0.4 nm", "illumination":metrics.get("illumination_scope"),
        "camera_probability":metrics["camera_collected_zero_loss_relative_intensity"],"flux_ledger":metrics["camera_flux_ledger"],
        "scope":"complete production path, synthetic linear permeability/NI; no instrument calibration, coarse ray integration for smoke test"}))
