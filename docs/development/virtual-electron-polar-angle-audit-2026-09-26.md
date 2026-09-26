# Virtual-electron polar angle and apparent deflection — 2026-09-26

## Finding and scope

The reported sub-mrad launch produces a real downstream displacement in the
current optical model, while the drawing strongly exaggerates its slope.
No mrad/degree conversion error or extra physical scaling was found.

Reconstruction follows MainWindow startup: default state, selected assembly,
nanoprobe/diffraction operating presets, physical layout and ideal mode. The
electron starts at XYZ=0 with 0.3 eV, polar=0.866371 mrad, azimuth=0 degrees,
maximum travelled path=3.0264 m. Projection is 75.4 degrees. This reproduces
the screenshot's visible 9,512 steps, Z=3026.392986 mm, TOF=14193.718 ps and
final kinetic energy=300000.3 eV. The running application's full state was not
recovered; the agreement is with its visible numerical signature and startup
configuration. Bare `default_state()` lacks the applied operating presets and
is a different optical case.

The GUI converts 0.866371 mrad to 0.0496394017925 degrees; the solver converts
that to 0.000866371 radians. The initial unit direction is
`(0.000866370891617, 0, 0.999999624701)`. Propagation retains full physical XYZ.
Display projection uses `U = X cos(projection) + Y sin(projection)` and never
feeds enlarged drawing coordinates into integration.

## Numerical evidence

At fixed assembly, fields, tip energy/position, azimuth and path limit:

| Initial polar angle (mrad) | Final radial displacement (mm) |
| ---: | ---: |
| 0 | 0 |
| 0.1 | 0.140501 |
| 0.5 | 0.702481 |
| 0.866371 | 1.216858 |
| 1 | 1.404788 |

The near-axis response is approximately linear over this declared interval.
The on-axis launch remains on axis. At 0.866371 mrad, the final projected U
displacement is 1.176961 mm and the final 3D direction angle is 1.813557 mrad
(about 0.104 degrees). The final projected angle is 1.754102 mrad.

With the screenshot's effective transverse display ratio of approximately 294,
the apparent final projected slope is
`atan(294 × tan(0.001754102)) ≈ 27.3 degrees`. This explains a steep-looking
line without a corresponding tens-of-degrees physical deflection.

Acceleration initially reduces the direction angle: at Z400 mm the energy is
300000.3 eV, radial displacement 0.188806 micrometres and direction angle
0.000102735 mrad. The subsequent lens chain changes both beam position and
direction; around the specimen waist the radius is 8.42 nm and angle 13.009 mrad.
The launch polar angle is an initial condition, not a direction constraint at
every later plane.

Compiled and Python reference paths agree to a maximum sampled position
difference of 1.64e-13 m. Smaller steps and tighter tolerances produce final
radius 1.213305 mm, versus 1.216858 mm at the normal live settings: a 0.293%
difference. This supports the qualitative displacement and rules out a large
numerical artifact in this case; it does not establish exact global convergence
or full-instrument physical qualification. No optical model, field strengths,
emission model, or integration tolerances were changed by this UI repair.

## Controls and explanation

- The polar slider now covers 0–5 mrad in 0.0025 mrad increments. The number
  editor steps by 0.01 mrad. Explicit larger diagnostic angles and existing
  records remain readable without silently truncating them to the slider range.
- **Polar angle** is the tilt from +Z. **Azimuth** is the direction of that
  tilt around +Z, measured from +X towards +Y. At zero tilt, azimuth does not
  affect the direction. It is separate from the display projection angle.
- The existing energy readout adds the physical initial/final 3D direction
  angles in mrad. A tooltip provides the maximum along the path; zero velocity
  reports an undefined direction. Partial results are explicitly labelled live.
- The field-plot footer states **angles not to scale**.

The coordinate convention agrees with Richard Fitzpatrick's
[Spherical Coordinates](https://farside.ph.utexas.edu/teaching/336L/Fluid/node258.html).

## Validation and local evidence

83 controller/continuous-update checks and 85 canvas/navigation checks pass
(168 total, zero failures/errors/skips). The narrow 360 px dock remains free of
horizontal overflow. Compilation and whitespace checks pass.

- `tmp/polar-angle-gui-final.xml`
- `tmp/polar-angle-display-final.xml`
- `tmp/virtual-electron-gui-default-small-angle-audit.json`
- `tmp/virtual-electron-small-angle-audit.json` (separate bare-default case)
- `tmp/continuous-electron-mrad-dock-360.png`

Generated numerical reports remain local. Restart the application for the
new fine slider, direction readout and display explanation.
