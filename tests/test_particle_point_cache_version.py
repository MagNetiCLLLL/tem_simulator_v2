"""Do not reuse spectra formed at the legacy unresolved/default point."""

import temsim.calculation_cache as cache
from temsim.optics.column import default_state


def test_point_change_versions_particle_products_without_discarding_wave_or_column(monkeypatch):
    state = default_state()
    current = cache.calculation_signatures(state)
    monkeypatch.setattr(cache, "_PARTICLE_POINT_SCHEMA", "legacy-unresolved-point")
    previous = cache.calculation_signatures(state)
    changed = {key for key in current if current[key] != previous[key]}
    assert changed == {"request", "elastic", "eds", "sample_region"}
    assert {"incident", "column", "stem", "wave", "wave_source"} <= cache.matching_products(previous, current)
