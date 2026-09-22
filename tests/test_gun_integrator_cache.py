"""Executed gun numerics invalidate both gun and downstream calculation caches."""
import pytest

from temsim.calculation_cache import calculation_signatures
from temsim.optics.column import default_state
from temsim.optics.electron_gun import tracing


@pytest.mark.parametrize("name,value", [
    ("ANALYTIC_STEP_SCHEMA", "test-different-integrator"),
    ("ANALYTIC_MAXIMUM_RELATIVE_IMPULSE", .0125),
])
def test_gun_numerics_invalidate_executed_transport(monkeypatch, name, value):
    state = default_state()
    gun = state.electron_gun
    before_gun = gun._cache_key(9)
    before = calculation_signatures(state)
    monkeypatch.setattr(tracing, name, value)
    after = calculation_signatures(state)
    assert gun._cache_key(9) != before_gun
    for product in ("request", "incident", "column", "elastic", "eds", "stem"):
        assert after[product] != before[product], product
