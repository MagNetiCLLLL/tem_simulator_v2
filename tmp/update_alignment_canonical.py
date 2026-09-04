from pathlib import Path
p=Path("src/temsim/optics/direct_alignment.py")
s=p.read_text()
s=s.replace("from temsim.physics.core import E, fields, propagate", "from temsim.physics.core import E, fields, propagate, interleaved_rk4_values\nfrom temsim.physics.ray_integrator import _canonical_step_numba")
a=s.index("@njit(cache=True)\ndef _rk4_transfer_matrix(")
b=s.index("@njit(cache=True)\ndef _rk4_axisymmetric_larmor_matrix(",a)
s=s[:a]+'''@njit(cache=True)
def _rk4_transfer_matrices(g, sx, sy, z_m, capture_indices):
    """Map laboratory slopes with the production canonical RK4 stages.

    Coefficients interleave exact endpoints and midpoints.  Keeping the
    affine-free four bases in slope coordinates at each node also handles
    source and capture planes inside a magnetic lens correctly.
    """
    captured = np.empty((capture_indices.size, 4, 4), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    capture = 0
    if capture_indices.size and capture_indices[0] == 0:
        captured[0] = matrix
        capture = 1
    for index in range(z_m.size - 1):
        h = z_m[index + 1] - z_m[index]
        a, b, c = 2 * index, 2 * index + 1, 2 * index + 2
        for column in range(4):
            x, tx, y, ty = _canonical_step_numba(
                matrix[0, column], matrix[2, column],
                matrix[1, column], matrix[3, column], h,
                g[a], g[b], g[c],
                sx[a], sx[b], sx[c], sy[a], sy[b], sy[c],
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            )
            matrix[0, column], matrix[1, column] = x, y
            matrix[2, column], matrix[3, column] = tx, ty
        while capture < capture_indices.size and capture_indices[capture] == index + 1:
            captured[capture] = matrix
            capture += 1
    return captured


@njit(cache=True)
def _rk4_transfer_matrix(g, sx, sy, z_m):
    return _rk4_transfer_matrices(
        g, sx, sy, z_m, np.asarray((z_m.size - 1,), dtype=np.int64)
    )[0]


'''+s[b:]
s=s.replace("        q0 = g[index] * g[index]\n        midpoint_g = 0.5 * (g[index] + g[index + 1])\n        qm = midpoint_g * midpoint_g\n        q1 = g[index + 1] * g[index + 1]\n", "        q0 = g[2 * index] * g[2 * index]\n        qm = g[2 * index + 1] * g[2 * index + 1]\n        q1 = g[2 * index + 2] * g[2 * index + 2]\n",1)
s=s.replace("    magnetic_t, sx_m2, sy_m2 = fields(z_mm, state)\n", "    stage_z_mm = interleaved_rk4_values(\n        z_mm, 0.5 * (z_mm[:-1] + z_mm[1:])\n    )\n    magnetic_t, sx_m2, sy_m2 = fields(stage_z_mm, state)\n",1)
s=s.replace("            0.5 * (g[1:] + g[:-1]) * np.diff(z_m)\n", "            (g[:-2:2] + 4.0 * g[1::2] + g[2::2])\n            * np.diff(z_m) / 6.0\n",1)
s=s.replace("        self.z_m = self.z_mm * 1.0e-3\n", "        self.z_m = self.z_mm * 1.0e-3\n        self.stage_z_mm = interleaved_rk4_values(\n            self.z_mm, 0.5 * (self.z_mm[:-1] + self.z_mm[1:])\n        )\n",1)
s=s.replace("fields(self.z_mm, state)","fields(self.stage_z_mm, state)")
s=s.replace("            np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2),\n", "            np.zeros(3), np.zeros(3), np.zeros(3),\n",1)
s=s.replace("            np.zeros(2), np.array((0.0, 1.0))\n", "            np.zeros(3), np.array((0.0, 1.0))\n",1)
s=s.replace("    def _field_arrays(self, vector) -> tuple[np.ndarray, np.ndarray]:", "    def _field_arrays(self, vector) -> np.ndarray:",1)
s=s.replace("        dg = np.ascontiguousarray(\n            np.gradient(g, self.z_m, edge_order=1), dtype=np.float64\n        )\n        return g, dg", "        return g",1)
s=s.replace("g, dg = self._field_arrays(vector)", "g = self._field_arrays(vector)")
s=s.replace("g, _dg = self._field_arrays(vector)", "g = self._field_arrays(vector)")
s=s.replace("            g, dg, self.sx_m2, self.sy_m2, self.z_m", "            g, self.sx_m2, self.sy_m2, self.z_m",1)
s=s.replace("            g,\n            dg,\n            self.sx_m2,", "            g,\n            self.sx_m2,",1)
p.write_text(s,encoding="utf-8")
