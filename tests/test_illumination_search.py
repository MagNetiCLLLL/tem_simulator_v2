"""Fast isolated branch-search fixtures, not full-chain optical acceptance."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.illumination_search import focus_grid_proposals


def test_focus_grid_proposes_distinct_branches_without_changing_optics(monkeypatch):
    import temsim.optics.direct_alignment as alignment
    import temsim.optics.electron_gun.source as source_module
    phi = np.linspace(0, 2*np.pi, 32, endpoint=False)
    bundle = SimpleNamespace(x_m=np.cos(phi)*1e-9, y_m=np.sin(phi)*1e-9,
        tx_rad=np.cos(phi)*.03, ty_rad=np.sin(phi)*.03,
        alive=np.ones(32,bool), weight=np.ones(32)/32)
    monkeypatch.setattr(source_module, 'trace_source_to_exit', lambda s:SimpleNamespace(exit_bundle=bundle))
    class Map:
        vector_maps = False
        upper = np.array([100.,100.])
        def __init__(self,*args,**kwargs):
            pass
        def matrices_at(self, vector, planes):
            # Deliberately synthetic lens: covariance root at objective = 50,
            # slope orthogonal to position there (finite nonzero emittance).
            primary, objective = vector
            matrix = np.zeros((4,4))
            matrix[0,0] = matrix[1,1] = 1
            matrix[2,0] = matrix[3,1] = (objective-50)*1e6
            matrix[2,3], matrix[3,2] = -primary/50, primary/50
            return np.array([matrix]*len(planes))
    monkeypatch.setattr(alignment,'_LiveFirstOrderModel',Map)
    state = SimpleNamespace(electron_gun=SimpleNamespace(exit_plane_z_mm=1),
        sample=SimpleNamespace(z_mm=100,thickness_nm=10),apertures=[],
        nanopulser=SimpleNamespace(installed=False))
    proposals, audit = focus_grid_proposals(state, ('c1','objective_lens'), 'nano_probe',
        primary_points=9,objective_points=5,keep=3)
    assert len(proposals)==3
    assert all(p.values[1]==pytest.approx(50) for p in proposals)
    assert proposals[0].score <= proposals[1].score <= proposals[2].score
    assert audit['qualification']=='PROPOSALS_ONLY'
    assert audit['focus_roots']>0
    assert state.sample.z_mm==100


@pytest.mark.parametrize('options', [dict(primary_points=2),dict(objective_points=2),dict(keep=0)])
def test_invalid_search_dimensions_fail_before_executing_a_gun(options):
    with pytest.raises(ValueError,match='search dimensions'):
        focus_grid_proposals(None,('c1','objective_lens'),'nano_probe',**options)
