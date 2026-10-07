"""One retained mesh model shared by saved-assembly presentation pages."""

from temsim.assembly_model_3d import (
    assembly_model_fingerprint, assembly_model_from_assembly,
)


class AssemblyGeometryCache:
    """Build on demand; view angles never participate in geometry identity.

    A single entry bounds memory and avoids rebuilding the same assembly when
    moving between the 3D and rotating-section pages. Cell context is composed
    by each page, so changes to a window do not rebuild column hardware.
    """

    def __init__(self, *, angular_segments=32):
        self.angular_segments = angular_segments
        self._fingerprint = None
        self._model = None
        self.mesh_builds = 0

    def model_for(self, assembly, runtime_values=None):
        fingerprint = (
            self.angular_segments,
            assembly_model_fingerprint(assembly, runtime_values),
        )
        if fingerprint != self._fingerprint:
            model = assembly_model_from_assembly(
                assembly, runtime_values=runtime_values,
                angular_segments=self.angular_segments,
            )
            self._fingerprint, self._model = fingerprint, model
            self.mesh_builds += 1
        return self._model
