"""A recording-model update must not relabel older STEM products current."""

import temsim.calculation_cache as cache
from temsim.optics.column import default_state


def test_recording_model_change_invalidates_stem_without_losing_upstream_work(monkeypatch):
    state = default_state()
    current = cache.calculation_signatures(state)
    monkeypatch.setattr(cache, "_STEM_RECORDING_SCHEMA", "legacy-no-downstream-kicks")
    previous = cache.calculation_signatures(state)

    changed = {key for key in current if current[key] != previous[key]}
    assert changed == {
        "request", "stem", "stem_transport", "fourdstem_cube",
        "fourdstem_virtual_detectors", "fourdstem_physical_recording",
    }
    reusable = cache.matching_products(previous, current)
    assert {"incident", "column", "elastic", "eds", "wave", "wave_source"} <= reusable
    assert not reusable.intersection(changed)
