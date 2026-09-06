# Magnetic-field validation and R-Z inspection

Implementation checkpoint: 2026-09-05. This is a numerical study of a configured
axisymmetric magnetic model, not an instrument calibration or TEM/STEM image.

## Workflow

1. Open **Simulation > Configure lens models...**. Select an enabled round lens
   with a saved linear-geometry or nonlinear B-H recipe. Analytic/Ideal tiers do
   not activate retained geometry recipes on the user's behalf.
2. Open **Field validation**. Set **Target change**, **Refinement** and the local
   collimated **Test radius**. Select **Check mesh and boundary**.
3. Review **Convergence**: values, absolute changes, allowed changes and statuses
   are listed separately. The overlaid Bz curves use identical physical planes.
4. Open **Magnetic R-Z** to inspect the final study case, not a replacement of the
   active production field. Select field strength or material regions. White
   contours show poloidal magnetic flux per radian, `psi = R*A_phi`. The material
   view uses the same masks and split Objective intervals as the FEM solver.
5. **Export validation...** writes the displayed study's original geometry,
   numerical settings, current scales, B-H snapshots/source hashes, voltage,
   reference planes, curves, residuals and comparisons to JSON. A retained older
   study remains explicitly identified as such after inputs change.

The check never changes excitation, geometry, operating presets, ray counts,
sample interactions, wave images or production mesh settings. Select/apply any
production parameter change explicitly. All comparisons run on detached state
snapshots, sharing only immutable resolved assembly data.

## What is tested

The five cases are:

| Case | Mesh | Boundary |
| --- | --- | --- |
| Mesh 1/3 | Configured base mesh | Configured padding |
| Mesh 2/3 | Base interval counts multiplied by the refinement factor | Unchanged |
| Mesh 3/3 | Base interval counts multiplied by the squared factor | Unchanged |
| Boundary 1/2 | Retain every finest interior grid node; extend outward | Padding multiplied by the factor |
| Boundary 2/2 | Retain the previous interior grid; extend outward again | Padding multiplied by the squared factor |

Both radial and axial meshes are refined. Authoritative material edges remain
in the mesh. Expansion must not translate, stretch or redistribute the original
interior grid: otherwise the boundary comparison would also measure changed
interior discretization. Newly included passive material enters the expanded
domain. An active foreign coil that a nonlinear expansion would omit is rejected.

Each linear shared circuit sums its channel-owned basis responses once. A B-H
study solves all configured nonlinear channels jointly at their complete current
vector. The current vector and physical model stay fixed across the five cases.
Linear studies represent the selected circuit response, including passive
neighbours; they are not full-column sums of all independently driven lenses.

The comparisons use a common Z grid within the original physical reference
interval. Uniform points, original FEM nodes and their midpoints are retained;
near-duplicate floating-point coordinates are merged. Differences are sampled
estimates, not rigorous global error bounds or a Richardson extrapolation.

### Field and local paraxial observables

- Bz curve: maximum absolute difference over the common axial grid.
- Effective focal length: `-1/C` from the finite-segment Larmor-frame transfer
  matrix, in mm; not a fitted Gaussian or an adjusted preset. It is a local
  paraxial focusing metric. End planes are fixed and not certified field-free;
  it must not be substituted for a calibrated whole-lens or objective-image
  focal length without checking those assumptions.
- Signed Larmor rotation: `integral(-q*Bz/(2*p)) dz`, in degrees; electron charge
  q is negative, momentum p is relativistic, and positive rotation is about +Z.
- Exit test-beam radius: `abs(A)*input_radius` for a collimated paraxial bundle,
  in um. This is not the user's source beam, specimen output or detector image.

The rotating-frame equation is `u'' + g(z)^2*u = 0`, where
`g = -q*Bz/(2*p)` and primes denote derivatives with respect to Z in metres.
The two fundamental solutions start at `(u,u')=(1,0)` and `(0,1)`.
Piecewise-linear Bz intervals are integrated explicitly with RK4. Subdivision
is doubled through 2, 4, 8 and 16 until successive transfer entries agree within
`rtol=1e-8, atol=1e-10`. The unit-determinant check uses 1e-5. A failure is
reported, never silently accepted. Test bundles exceeding 100 mrad or sampling
magnetic material/coil regions are rejected. Physical aperture and specimen
interception are outside this diagnostic's scope.

No empirical Cs or Cc is added to this local test. **Cs/Cc validation remains
Not checked**: axial-field/near-axis convergence does not prove stability of
high-order off-axis derivatives, finite-pupil fits or wave-image aberrations.

### Acceptance and unavailable values

For every successive comparison:

`absolute_change <= absolute_floor + relative_target*abs(refined_value)`

For a field curve, the refined scale is its maximum absolute Bz. The default
relative target is 1%; it is a configurable engineering target, not certified
accuracy. Absolute floors are 1 uT for Bz, 10 nm for focal length, 0.0001 deg
for rotation and 0.0001 um for test radius. The Python options expose all floors.

Zero/near-zero focusing power has no finite focal length and is **Unavailable**,
not infinity or a forced zero. Relative change is unavailable at zero refined
magnitude; absolute checks still apply. Overall **Sampled checks within
tolerance** requires all listed comparisons to pass. It is not a whole-column,
material-data, Cs/Cc, experimental or independent-software validation claim.

## Isolation, reuse and limits

- A separate background job reports completed cases out of five. One case may
  contain multiple linear basis solves. There is no elapsed-time-based progress.
- Cancellation is cooperative between field solves. Sparse/Newton work already
  in progress finishes its current solve. Cancellation/failure preserves previous
  reports and all production images; closing the window rejects late results.
- Completed studies are retained in a four-entry dependency-keyed LRU. Selected
  circuits, relevant geometry/neighbours, excitation, materials, voltage, model
  and study options are fingerprinted. Sample/image settings and a distant
  unrelated linear lens do not invalidate the selected circuit's study. All B-H
  channels are jointly dependent, as are active foreign coils in their domain.
- Returning to retained inputs reuses the report. A worker finishing for older
  inputs can populate this cache but cannot overwrite the current view.
- Base maps can reuse the production field cache. Refined maps use a separate
  twelve-entry cache, so they do not evict production field maps. Mesh preparation
  is shared. Plot layers, tabs, zoom and pan never request another field solve.
- Base node-count/refinement limits, padding <=10 and a 250000-node explicit
  validation-grid limit fail explicitly. No automatic mesh/current clamping.
- Drawing samples are 180 radial by 320 axial points. They are for inspection,
  not a replacement of the underlying solver mesh or a numerical acceptance test.

## References and remaining work

The [FEMM FAQ](https://www.femm.info/doku/doku.php?id=FAQ) discusses B-H
interpolation/extrapolation and mesh sensitivity. The
[pyFEMM interface](https://www.femm.info/doku/doku.php?id=pyFEMM) is a possible
independent comparison route. No FEMM installation, automated invocation or
FEMM-vs-TEM comparison was performed here. The export prepares reproducible input
and output data, not a FEMM-native model file or a passed external benchmark.
Curve interpolation and outer boundary conditions must be aligned or explicitly
accounted for in such a comparison.

Next: independent field benchmarks, high-order Cs/Cc convergence, and calibrated
material/lens data. Thermal, hysteretic and arbitrary-3D coupling remain outside
this stage. Do not retune presets to make numerical comparisons agree.
