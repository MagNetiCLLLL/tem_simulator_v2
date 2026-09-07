"""Read-only project review repro; writes only inside TemporaryDirectory."""
from tempfile import TemporaryDirectory

import numpy as np

from temsim.artifact_store import ArtifactStore, ArtifactTooLargeError
from temsim.calculation_manifest import capture_calculation_manifest
from temsim.optics.column import default_state


state = default_state()
first = capture_calculation_manifest(state, ray_count=25, step_mm=0.1)
with TemporaryDirectory(prefix="temsim-review-cache-") as root:
    # 1 MiB is accepted by validate_cache_preferences as a disk-cache budget.
    store = ArtifactStore(root, quota_bytes=1024**2)
    first_args = dict(
        product_key="incident",
        dependency_signature=first.calculation_signatures["incident"],
    )
    store.put_array_bundle(first, arrays={"values": np.arange(8)}, **first_args)
    print("Before oversized write:", len(list(store.references_root.glob("*.json"))), "reference(s)")
    state.lenses[0].percent += 1
    second = capture_calculation_manifest(state, ray_count=25, step_mm=0.1)
    try:
        store.put_array_bundle(
            second,
            product_key="incident",
            dependency_signature=second.calculation_signatures["incident"],
            arrays={"values": np.arange(262_144, dtype=np.int64)},
        )
    except ArtifactTooLargeError as exc:
        print(type(exc).__name__ + ":", exc)
    retained = store.get_array_bundle(first, **first_args) is not None
    print("After rejected write, original artifact retained:", retained)
    print("After oversized write:", len(list(store.references_root.glob("*.json"))), "reference(s)")
    if not retained:
        raise SystemExit("BUG reproduced: rejected write removed the previous valid cache entry")
