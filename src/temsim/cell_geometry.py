"""Read-only cell surfaces from the same applied layers used by particles.

Coordinates are column X/Y/Z in mm, +Z downstream. Membranes are flat finite
disks. Fluid and Sample-reference surfaces are context, not extra solids or
apertures. No geometry here is fed back into the optical assembly.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np

from temsim.vacuum import resolve_cell_layers

CELL_SAMPLE_KEY = "cell_sample_reference"
COLOURS = {"specimen_cell": (0.09, .6, .7, .12),
           "cell_window_upstream": (.25, .85, .95, .65),
           "cell_window_downstream": (.65, .5, .98, .65),
           CELL_SAMPLE_KEY: (1., .75, .15, .35)}


@dataclass(frozen=True)
class CellPhysicalContext:
    layers: tuple = ()
    sample_z_mm: float = 0.0
    sample_x_mm: float = 0.0
    sample_y_mm: float = 0.0

    @classmethod
    def capture(cls, state):
        return cls(deepcopy(resolve_cell_layers(state)), float(state.sample.z_mm),
                   state.sample.centre_x_nm*1e-6, state.sample.centre_y_nm*1e-6)

    def signature(self):
        if not self.layers:
            return "no-cell"
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def parts(self):
        parts = [SimpleNamespace(key=r.key, name=r.name, center_z_mm=(r.start_z_mm+r.end_z_mm)/2,
            start_z_mm=r.start_z_mm, end_z_mm=r.end_z_mm, radius_mm=r.radius_mm,
            center_x_mm=r.center_x_mm, center_y_mm=r.center_y_mm,
            data={"parent_key": "specimen_cell" if r.key != "specimen_cell" else None,
                  "material": r.medium.formula, "density_kg_m3": r.medium.density_kg_m3,
                  "cell_context": True}, source_file="Vacuum map / cell") for r in self.layers]
        if parts:
            parts.append(SimpleNamespace(key=CELL_SAMPLE_KEY, name="Sample reference (not material)",
                center_z_mm=self.sample_z_mm, start_z_mm=self.sample_z_mm, end_z_mm=self.sample_z_mm,
                data={"parent_key": "specimen_cell", "cell_context": True}, source_file="Sample"))
        return tuple(parts)

    def meshes(self, angular_segments=32):
        from temsim.part_model_3d import TriangleMesh, revolve_section
        from temsim.assembly_model_3d import _global_mesh
        meshes = []
        for r in self.layers:
            fluid = r.key == "specimen_cell"
            description = ("Cell interior boundary, not a solid; Sample volume excluded in transport" if fluid
                           else f"Flat membrane: {r.medium.formula}; {r.medium.density_kg_m3:g} kg/m3")
            # Build near local Z=0 to preserve sub-nm faces, then translate once.
            length = r.end_z_mm-r.start_z_mm
            mesh = revolve_section(((0, 0), (0, r.radius_mm), (length, r.radius_mm), (length, 0)),
                key=r.key, angular_segments=angular_segments, material_class=r.medium.formula,
                description=description)
            vertices = mesh.vertices.copy()
            vertices[:, :2] += (r.center_x_mm, r.center_y_mm)
            edges = ()
            if fluid:
                angles = np.linspace(0, 2*np.pi, angular_segments+1)
                rings = [np.column_stack((r.center_x_mm+r.radius_mm*np.cos(angles),
                         r.center_y_mm+r.radius_mm*np.sin(angles), np.full(angles.size, z))) for z in (0., length)]
                edges = tuple({"id": f"rim{i}", "vertices": ring} for i, ring in enumerate(rings))
                edges += tuple({"id": f"side{i}", "vertices": np.array([rings[0][i], rings[1][i]])}
                               for i in range(0, angular_segments, max(1, angular_segments//4)))
            meshes.append(_global_mesh(replace(mesh, vertices=vertices, color=COLOURS[r.key],
                                               edges=edges, wireframe=fluid), r.start_z_mm))
        if self.layers:
            radius = next(r.radius_mm for r in self.layers if r.key == "specimen_cell")
            phi = np.arange(angular_segments)*2*np.pi/angular_segments
            vertices = np.column_stack((np.r_[0, radius*np.cos(phi)]+self.sample_x_mm,
                                        np.r_[0, radius*np.sin(phi)]+self.sample_y_mm,
                                        np.full(angular_segments+1, self.sample_z_mm)))
            faces = np.array([(0, i+1, (i+1) % angular_segments+1) for i in range(angular_segments)])
            meshes.append(_global_mesh(TriangleMesh(vertices, faces, CELL_SAMPLE_KEY,
                color=COLOURS[CELL_SAMPLE_KEY], material_class="No material",
                description="Sample reference outline from Sample; visible through solids, not another specimen",
                wireframe=True, edges=({"id": "sample_rim", "vertices": np.vstack((vertices[1:], vertices[1]))},)), 0.))
        return tuple(meshes)
