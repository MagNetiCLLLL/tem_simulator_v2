# Canonical path phase and exact incident reuse

Status: bounded numerical implementation, 2026-09-11. This does not qualify
the complete gun/specimen/detector chain or the full caustic domain.

## Coordinate and operator conventions

The state is `(x, y, p_x/p0, p_y/p0)` in metres and dimensionless canonical
momentum. Let `k = 2 pi / wavelength` and use the Weyl displacement

```text
W(q,p) f(x) = exp(i k p . (x - q/2)) f(x - q)
T = exp(i k s) W(d) U(M)
```

`M` must be symplectic in this declared basis. `s` is a scalar action with
units of metres. An endpoint ray matrix does not identify the metaplectic
lift: a full oscillator turn has the identity ray matrix but can change a
coherent field's sign. Physical paths therefore retain both scalar action
and a continuous lift; a bare matrix only selects a principal lift.

For consecutive operators, with `v = M2 d1`, the extra Weyl action is
`(d2_p . v_q - d2_q . v_p)/2`. The local-centre carrier used by `PlaneWave`
adds the conversion action

```text
(p' . q' - p . q)/2 + d_p . q' + d_p . d_q/2
(q',p') = M (q,p)
```

This scalar is retained in the complex envelope. Quadratic curvature and
linear tilt remain analytic carriers. No calculated amplitude is fitted to a
Gaussian, renormalized to repair an error, or globally rephased by comparison
with a numerical reference.

## Following the lift

The auxiliary positive-Gaussian frame is `A + i B / length`. Its unwrapped
determinant phase determines the continuous lift. The reference length is a
numerical chart only, not a source width or a new illumination input. Each
increment must have resolved reference-frame eigenphases below `pi/2`.
Unresolved increments and inconsistent endpoint phases are rejected before
committing the path state.

The angular-spectrum and Collins-integral numerical charts supply analytic
reference phases. Their differences from the recorded physical lift fix the
kernel prefactor and branch. The upstream producer advances the lift at every
field-integration step, including steps between retained apertures/boundaries.
Unsupported mixed-conjugacy sampling still fails explicitly; these tests are
not a proof for every astigmatic caustic or nonlinear optical component.

## Reuse without changing history

Each checkpoint keeps its complete captured snapshot. Independently, its
incident stage signature hashes the actual field/kick arrays, physical
aperture states, sampled bores, source, solver implementation and non-specimen
external inputs. These fields are resolved from all providers, so a nominally
downstream lens can still invalidate the upstream cache through its field.

The operator ends at the finite specimen's upper face. Changing composition
without changing that face or fields can reuse its coherent modes. A reused
result has a new owner snapshot and records the original execution snapshot,
parent checkpoint and matching stage signature. Old arrays and snapshots are
unchanged. A changed thickness, Z, shared field, aperture or numerical step
does not qualify as that same incident operator. Matching nearby lens values
approximately is not implemented or claimed.

Warm requests still resolve and hash dependencies; they skip coherent-mode
propagation, not every operation. This is not yet a cache for the specimen or
downstream image adapters. Restoring an old full snapshot with missing/changed
external files remains blocked; read-only historical viewing remains available.

## Evidence scope

`tests/test_canonical_action.py` checks full complex fields against analytic
tilted/off-axis Gaussians and an independently evaluated full-field FFT for a
non-Gaussian wave, without phase fitting. It also checks affine composition,
inverse propagation, oscillator winding, reference-chart independence and
atomic rejection. These are isolated mathematical fixtures, not production
illumination entrances. The Gaussian comparison uses a 512-cell window and
L2 tolerance `2e-8`; the smaller 256-cell trial had unresolved boundary/carrier
sampling and was not accepted by relaxing this tolerance.

`tests/test_gun_wave_transport.py` uses the assembled quadratic gun path and
compares exact warm reuse with a separate complete cold execution of the
changed work point. It also replaces a Si CIF with Au at the same path and
checks that only the incident operator is reused with new specimen ownership.
This is not a comparison of Si and Au scattering or images.

## Primary method references

- de Gosson (2022), *Symplectic Radon Transform and the Metaplectic
  Representation*, including the metaplectic/Weyl and Gaussian conventions:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9222323/
- Ozorio de Almeida and Ingold (2013), *Metaplectic sheets and caustic
  traversals in the Weyl representation*:
  https://arxiv.org/abs/1309.5068

Consulted 2026-09-11. Method references do not calibrate the microscope or
validate the current production gun-to-image work domain.
