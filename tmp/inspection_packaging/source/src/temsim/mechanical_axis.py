"""Canonical source-referenced mechanical-axis validation primitives."""

from __future__ import annotations

from dataclasses import dataclass


_TOLERANCE_MM = 1.0e-9


@dataclass(frozen=True)
class MechanicalNestingPermission:
    child_key: str
    parent_key: str
    reason: str

    def __post_init__(self):
        if not self.child_key or not self.parent_key:
            raise ValueError("Mechanical nesting keys must not be empty.")
        if self.child_key == self.parent_key:
            raise ValueError("A mechanical component cannot contain itself.")
        if not str(self.reason).strip():
            raise ValueError(
                "Mechanical nesting requires an explicit reason."
            )


@dataclass(frozen=True)
class ResolvedMechanicalPlacement:
    key: str
    start_z_mm: float
    center_z_mm: float
    end_z_mm: float
    nested_parent_key: str | None = None
    nesting_reason: str = ""

    def __post_init__(self):
        if self.end_z_mm < self.start_z_mm:
            raise ValueError(
                f"{self.key} mechanical envelope has a negative length."
            )


@dataclass(frozen=True)
class ResolvedMechanicalClearance:
    upstream_key: str
    downstream_key: str
    clearance_mm: float


@dataclass(frozen=True)
class ResolvedMechanicalAxis:
    placements: tuple[ResolvedMechanicalPlacement, ...]
    axis_order: tuple[str, ...]
    clearances: tuple[ResolvedMechanicalClearance, ...]

    @property
    def by_key(self):
        return {placement.key: placement for placement in self.placements}

    def clearance_between(self, upstream_key, downstream_key):
        for relation in self.clearances:
            if (
                relation.upstream_key == upstream_key
                and relation.downstream_key == downstream_key
            ):
                return relation.clearance_mm
        raise KeyError((upstream_key, downstream_key))


def _placement_from_component(component, permission):
    center = float(component.mechanical_center_from_tip_mm)
    length = float(component.mechanical_length_mm)
    if length < 0.0:
        raise ValueError(
            f"{component.key} mechanical length must not be negative."
        )
    return ResolvedMechanicalPlacement(
        key=str(component.key),
        start_z_mm=center - 0.5 * length,
        center_z_mm=center,
        end_z_mm=center + 0.5 * length,
        nested_parent_key=(
            permission.parent_key if permission is not None else None
        ),
        nesting_reason=(
            permission.reason if permission is not None else ""
        ),
    )


def resolve_mechanical_axis(
    components,
    axis_order,
    nesting_permissions=(),
):
    """Resolve one mechanical branch and reject every unapproved overlap."""

    components_by_key = {}
    for component in components:
        key = str(component.key)
        if key in components_by_key:
            raise ValueError(f"Duplicate mechanical component key: {key}")
        components_by_key[key] = component

    permissions = {}
    for permission in nesting_permissions:
        if permission.child_key in permissions:
            raise ValueError(
                "Duplicate mechanical nesting permission for "
                f"{permission.child_key}."
            )
        permissions[permission.child_key] = permission

    order = tuple(str(key) for key in axis_order)
    if len(set(order)) != len(order):
        raise ValueError("Mechanical axis order contains duplicate keys.")
    unknown_order = set(order) - components_by_key.keys()
    if unknown_order:
        raise ValueError(
            "Mechanical axis contains unknown components: "
            + ", ".join(sorted(unknown_order))
        )
    unknown_nested = set(permissions) - components_by_key.keys()
    if unknown_nested:
        raise ValueError(
            "Mechanical nesting contains unknown children: "
            + ", ".join(sorted(unknown_nested))
        )
    for permission in permissions.values():
        if permission.parent_key not in components_by_key:
            raise ValueError(
                f"Unknown mechanical nesting parent: {permission.parent_key}"
            )
        if permission.child_key in order:
            raise ValueError(
                f"Nested component {permission.child_key} must not also "
                "consume serial axis space."
            )

    unowned = (
        components_by_key.keys() - set(order) - set(permissions)
    )
    if unowned:
        raise ValueError(
            "Every mechanical component must be serial or explicitly nested: "
            + ", ".join(sorted(unowned))
        )

    placements = tuple(
        _placement_from_component(
            component,
            permissions.get(str(component.key)),
        )
        for component in components_by_key.values()
    )
    by_key = {placement.key: placement for placement in placements}

    for permission in permissions.values():
        child = by_key[permission.child_key]
        parent = by_key[permission.parent_key]
        if (
            child.start_z_mm < parent.start_z_mm - _TOLERANCE_MM
            or child.end_z_mm > parent.end_z_mm + _TOLERANCE_MM
        ):
            raise ValueError(
                f"Nested component {child.key} is not fully contained "
                f"by {parent.key}."
            )

    clearances = []
    for upstream_key, downstream_key in zip(order, order[1:]):
        upstream = by_key[upstream_key]
        downstream = by_key[downstream_key]
        clearance = downstream.start_z_mm - upstream.end_z_mm
        if clearance < -_TOLERANCE_MM:
            raise ValueError(
                "Unapproved mechanical overlap: "
                f"{upstream_key} -> {downstream_key} "
                f"({clearance:.9g} mm clearance)."
            )
        clearances.append(
            ResolvedMechanicalClearance(
                upstream_key,
                downstream_key,
                max(clearance, 0.0),
            )
        )

    for index, left in enumerate(placements):
        for right in placements[index + 1:]:
            overlap = min(left.end_z_mm, right.end_z_mm) - max(
                left.start_z_mm, right.start_z_mm
            )
            if overlap <= _TOLERANCE_MM:
                continue
            permission = None
            child = None
            parent = None
            if left.nested_parent_key == right.key:
                permission, child, parent = left, left, right
            elif right.nested_parent_key == left.key:
                permission, child, parent = right, right, left
            if permission is None:
                raise ValueError(
                    "Unapproved mechanical overlap: "
                    f"{left.key} with {right.key} ({overlap:.9g} mm)."
                )
            if (
                child.start_z_mm < parent.start_z_mm - _TOLERANCE_MM
                or child.end_z_mm > parent.end_z_mm + _TOLERANCE_MM
            ):
                raise ValueError(
                    f"Nested component {child.key} is not fully contained "
                    f"by {parent.key}."
                )

    return ResolvedMechanicalAxis(
        placements,
        order,
        tuple(clearances),
    )
