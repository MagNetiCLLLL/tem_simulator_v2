# Sample display regions

The Sample page separates physical geometry from the local atom display.

- **Blue outline — Full sample:** the complete user-defined disk or rectangular
  envelope, with its diameter/size and physical thickness. The outline is never
  omitted merely because the local calculation region is much smaller.
- **Pink boundary — Local calculation region:** the bounding box of the material
  intersection with the selected completed TEM/STEM wave window. Corners outside
  a circular specimen can contain vacuum; the atoms themselves are clipped to
  the circular boundary. Surrounding vacuum in the FFT window is not material.
- **Spheres:** equilibrium CIF atoms reconstructed inside that local region.
  If the rendering cap is reached, an **amber display-subset box** and a short
  dimension label identify the smaller rendered window. This does not change
  the physical sample or calculation domain.

`Fit full sample` and `Fit local region` only change the camera. They do not
generate atoms or recalculate rays, waves or detector signals. Redrawing the
scene preserves the user's view range. All geometry uses a common nanometre
coordinate system and an undistorted aspect ratio; a millimetre specimen and a
nanometre local region cannot both be resolved at atomic detail at the same zoom.

Completed-domain metadata comes from `specimen_wave_window_bounds_nm` in the
displayed STEM or TEM result; STEM takes precedence when both are present.
The full outline and local region use the same captured sample. A changed draft
is explicitly distinguished from that captured sample. Without completed-domain
metadata, the view says `Structural preview` or `Estimated scan region`; it does
not claim that a calculation has used that region.

The spheres are a structural reconstruction, not stored frozen-phonon atom
coordinates or a visualization of electron trajectories. Changing the display
limit does not modify the calculation. A missing or unreadable CIF suppresses
the spheres but retains valid known full/local geometry and reports the error
in the atom-status tooltip.

Crystal zone alignment rotates the lattice before the current atomistic kernel
clips it to its laboratory-frame finite envelope. The displayed envelope is not
rotated a second time by that crystallographic orientation. This matches the
current calculation convention; it is not a new tilted-solid transport model.
