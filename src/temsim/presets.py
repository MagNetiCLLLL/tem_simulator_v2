from temsim.component_keys import (
    ADAPTER_LENS,
    CONDENSER_LENS_1,
    CONDENSER_LENS_2,
    CONDENSER_LENS_3,
    DIFFRACTION_LENS,
    INTERMEDIATE_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
    MINI_CONDENSER,
    OBJECTIVE_LENS,
    PROBE_TL12_LENS,
    PROBE_TL21_LENS,
    PROBE_TL22_LENS,
)

TEM_ILLUMINATION={CONDENSER_LENS_1:60.0,CONDENSER_LENS_2:45.0,CONDENSER_LENS_3:40.0,ADAPTER_LENS:42.0,PROBE_TL22_LENS:45.0,PROBE_TL21_LENS:52.0,PROBE_TL12_LENS:100.0,MINI_CONDENSER:35.0,OBJECTIVE_LENS:100.0}
STEM_ILLUMINATION={CONDENSER_LENS_1:90.0,CONDENSER_LENS_2:25.0,CONDENSER_LENS_3:55.0,ADAPTER_LENS:42.0,PROBE_TL22_LENS:45.0,PROBE_TL21_LENS:52.0,PROBE_TL12_LENS:100.0,MINI_CONDENSER:10.0,OBJECTIVE_LENS:37.981544506600244}
IMAGE_PROJECTOR={DIFFRACTION_LENS:25.137,INTERMEDIATE_LENS:40.0,PROJECTOR_LENS_1:30.0,PROJECTOR_LENS_2:30.0}
DIFFRACTION_PROJECTOR={DIFFRACTION_LENS:26.216,INTERMEDIATE_LENS:40.0,PROJECTOR_LENS_1:30.0,PROJECTOR_LENS_2:30.0}
P={
"TEM image":{"illumination":"TEM","projector":"image","lens":{**TEM_ILLUMINATION,**IMAGE_PROJECTOR}},
"TEM diffraction":{"illumination":"TEM","projector":"diffraction","lens":{**TEM_ILLUMINATION,**DIFFRACTION_PROJECTOR}},
"STEM image":{"illumination":"STEM","projector":"image","lens":{**STEM_ILLUMINATION,**IMAGE_PROJECTOR}},
"STEM diffraction":{"illumination":"STEM","projector":"diffraction","lens":{**STEM_ILLUMINATION,**DIFFRACTION_PROJECTOR}},}
def apply(state,name):
    preset=P[name];state.illumination_mode=preset["illumination"];state.projector_mode=preset["projector"]
    strengths=preset["lens"]
    for lens in state.lenses:
        if lens.key in strengths:lens.percent=float(strengths[lens.key])
    state.electron_gun.electrostatic_lens.voltage_kv=1.2
    state.sync_objective()
