"""Emission starts at the physical emitter; computed checkpoints are caches."""


class UnsupportedSourceModel(ValueError):
    code = "TIP_ORIGIN_REQUIRED"


EXIT_SOURCE_REJECTION = (
    "Custom exit sources are not permitted. Electrons must originate at the tip "
    "and pass through extraction, acceleration, gun focusing and apertures. "
    "A downstream state may only cache the result of upstream component transport."
)


def require_physical_gun_source(gun):
    if getattr(gun, "source_representation", "classical_particles") != "classical_particles":
        raise UnsupportedSourceModel(EXIT_SOURCE_REJECTION)


def require_tip_coherent_source(gun):
    require_physical_gun_source(gun)
    raise UnsupportedSourceModel(
        "Complete coherent propagation from the tip to the specimen is not implemented. "
        "A separate development tip-to-exit quadratic solver is available, but cannot "
        "enable this historical exit-source path. Physical particle transport remains "
        "available; an effective exit source cannot replace it."
    )
