"""Transactional drafts of existing TOML dimensions and material regions."""

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real
from pathlib import Path
import tomllib

from temsim import module_manifest
from temsim.manifest_editor import resized_part_axial_coordinates
from temsim.component_operations import (
    PartChangeSet, added_component_document, placed_component_document,
    copied_component_document, validate_component_graph, CUSTOM_COMPONENT_ROLES,
)


class PartModelDocument:
    """A file-backed draft shared by every selected part in the 3D editor."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self._source_bytes = self.path.read_bytes()
        self._baseline = tomllib.loads(self._source_bytes.decode("utf-8-sig"))
        if not isinstance(self._baseline.get("parts"), list) or not self._baseline["parts"]:
            raise ValueError("Open an instrument module TOML containing [[parts]] definitions")
        keys = [part.get("key") for part in self._baseline["parts"]]
        if any(not isinstance(key, str) or not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("Every part must have a unique non-empty key")
        by_key = {part["key"]: part for part in self._baseline["parts"]}
        for key in keys:
            seen = set()
            ancestor = key
            while ancestor in by_key:
                if ancestor in seen:
                    raise ValueError("Component ownership must not contain a cycle")
                seen.add(ancestor)
                ancestor = by_key[ancestor].get("parent_key")
        self.document = deepcopy(self._baseline)
        self._history = [deepcopy(self.document)]
        self._history_index = 0

    @property
    def dirty(self):
        return self.document != self._baseline

    @property
    def can_undo(self):
        return self._history_index > 0

    @property
    def can_redo(self):
        return self._history_index + 1 < len(self._history)

    def part(self, key):
        return next(part for part in self.document["parts"] if part["key"] == key)

    def _remember(self):
        if self.document == self._history[self._history_index]:
            return
        self._history = self._history[:self._history_index + 1]
        self._history.append(deepcopy(self.document))
        if len(self._history) > 100:
            self._history.pop(0)
        self._history_index = len(self._history) - 1

    def _commit_part(self, key, candidate):
        for index, part in enumerate(self.document["parts"]):
            if part["key"] == key:
                self.document["parts"][index] = candidate
                self._remember()
                return
        raise ValueError(f"Unknown part: {key}")

    def _commit_component_operation(self, candidate):
        # Structural edits must be valid before replacing any part of the
        # current draft; validation failure preserves both draft and history.
        validate_component_graph(candidate)
        module_manifest.validate_document(candidate)
        self.document = candidate
        self._remember()

    def add_component(self, part):
        """Insert an independent mechanical part as one undoable operation."""
        candidate, key = added_component_document(self.document, part)
        self._commit_component_operation(candidate)
        return key

    def place_component(self, key, center_z_mm, include_children=True):
        """Translate module-local axial coordinates, optionally with children."""
        candidate, keys = placed_component_document(
            self.document, key, center_z_mm, include_children=include_children,
        )
        self._commit_component_operation(candidate)
        return keys

    def copy_component_from(self, source, key, new_key, center_z_mm,
                            parent_key=None, include_children=True, *, name=None):
        """Make an independent mechanical copy without adding optical controls."""
        candidate, new_key = copied_component_document(
            self.document, source, key, new_key, center_z_mm,
            parent_key=parent_key, include_children=include_children, name=name,
        )
        self._commit_component_operation(candidate)
        return new_key

    @staticmethod
    def _finite_dimension(value):
        try:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("A dimension must be a finite number")
            return float(value)
        except (OverflowError, TypeError) as exc:
            raise ValueError("A dimension must be a finite number") from exc

    @staticmethod
    def _numeric_target(root, path):
        if not path:
            raise ValueError("Choose an existing numeric dimension")
        current = root
        try:
            for segment in path[:-1]:
                if isinstance(current, list) and (
                    isinstance(segment, bool) or not isinstance(segment, int) or segment < 0
                ):
                    raise ValueError("Choose an existing array element")
                current = current[segment]
            final = path[-1]
            if isinstance(current, list) and (
                isinstance(final, bool) or not isinstance(final, int) or final < 0
            ):
                raise ValueError("Choose an existing array element")
            previous = current[final]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Choose an existing dimension path") from exc
        if isinstance(previous, bool) or not isinstance(previous, Real):
            raise ValueError("Choose an existing numeric dimension")
        return current, final

    @staticmethod
    def _resize_copied_axial_geometry(part, length):
        if part.get("mechanical_part_role") != "custom_mechanical_copy":
            return
        previous = float(part["length_mm"])
        if previous <= 0 or previous == length:
            return
        ratio = length / previous
        center = float(part["local_center_z_mm"])
        if "material_intervals_mm" in part:
            part["material_intervals_mm"] = [
                [center + (float(value) - center) * ratio for value in interval]
                for interval in part["material_intervals_mm"]
            ]
        if "magnetic_radial_profile_mm" in part:
            part["magnetic_radial_profile_mm"] = [
                [float(row[0]) * ratio, *row[1:]] for row in part["magnetic_radial_profile_mm"]
            ]

    @staticmethod
    def _synchronize_custom_base_length(part):
        base = part.get("model_3d", {}).get("base", {})
        if (part.get("mechanical_part_role") in CUSTOM_COMPONENT_ROLES
                and base.get("kind", "existing") != "existing"):
            length = float(base["length_mm"])
            PartModelDocument._resize_copied_axial_geometry(part, length)
            part.update(resized_part_axial_coordinates(part, length))
            part["length_mm"] = length

    def set_dimension(self, path, value):
        path = tuple(path)
        value = self._finite_dimension(value)
        if len(path) < 3 or path[0] != "parts":
            raise ValueError("Only existing part dimensions can be edited")
        part = deepcopy(self.part(path[1]))
        field = path[2]
        if field == "model_3d":
            from temsim.part_model_features import validate_model_3d

            current, final = self._numeric_target(part, path[2:])
            current[final] = value
            validate_model_3d(part)
            if path[2:] == ("model_3d", "base", "length_mm"):
                self._synchronize_custom_base_length(part)
            self._commit_part(path[1], part)
            return
        if field not in part or not field.endswith(("_mm", "_um", "_deg")):
            raise ValueError("Choose an existing dimension in millimetres or degrees")
        if field == "local_center_z_mm":
            raise ValueError("The axial centre is fixed; edit the existing length or endpoints")
        if len(path) == 3:
            if isinstance(part[field], bool) or not isinstance(part[field], Real):
                raise ValueError("Choose a numeric dimension")
            if field == "length_mm":
                endpoints = resized_part_axial_coordinates(part, value)
                self._resize_copied_axial_geometry(part, value)
                part.update(endpoints)
                base = part.get("model_3d", {}).get("base", {})
                if (part.get("mechanical_part_role") in CUSTOM_COMPONENT_ROLES
                        and base.get("kind", "existing") != "existing"):
                    base["length_mm"] = value
            part[field] = float(value)
            if field in {"local_start_z_mm", "local_end_z_mm"}:
                part["length_mm"] = float(part["local_end_z_mm"]) - float(part["local_start_z_mm"])
        else:
            current, index = self._numeric_target(part[field], path[3:])
            current[index] = float(value)
        self._commit_part(path[1], part)

    def set_model_3d(self, key, configuration):
        """Replace one mechanical CAD configuration, or remove it with None.

        Schema errors are rejected atomically. A schema-valid boolean operation
        may remain a draft until complete geometry is validated on Save.
        Existing physical dimensions and solver parameters are left unchanged.
        """
        from temsim.part_model_features import validate_model_3d

        candidate = deepcopy(self.part(key))
        if configuration is None:
            candidate.pop("model_3d", None)
        elif isinstance(configuration, Mapping):
            candidate["model_3d"] = deepcopy(dict(configuration))
        else:
            raise ValueError("model_3d must be a table or None")
        validate_model_3d(candidate)
        self._synchronize_custom_base_length(candidate)
        self._commit_part(key, candidate)

    def update_model_3d(self, key, updates):
        """Merge nested tables in one undo step; replace supplied collections."""
        from temsim.part_model_features import default_model_3d

        if not isinstance(updates, Mapping):
            raise ValueError("model_3d updates must be a table")
        part = self.part(key)
        configuration = deepcopy(part.get("model_3d", default_model_3d(part)))

        def merge(target, values):
            for field, value in values.items():
                if isinstance(value, Mapping) and isinstance(target.get(field), dict):
                    merge(target[field], value)
                else:
                    target[field] = deepcopy(value)

        merge(configuration, updates)
        self.set_model_3d(key, configuration)

    def upsert_model_feature(self, key, feature):
        """Add or replace a stable feature id without changing other features."""
        from temsim.part_model_features import default_model_3d

        if not isinstance(feature, Mapping) or not isinstance(feature.get("id"), str) or not feature["id"]:
            raise ValueError("A feature needs a non-empty stable id")
        part = self.part(key)
        configuration = deepcopy(part.get("model_3d", default_model_3d(part)))
        features = configuration.setdefault("features", [])
        for index, previous in enumerate(features):
            if previous.get("id") == feature["id"]:
                features[index] = deepcopy(dict(feature))
                break
        else:
            features.append(deepcopy(dict(feature)))
        self.set_model_3d(key, configuration)

    def remove_model_feature(self, key, feature_id):
        configuration = deepcopy(self.part(key).get("model_3d", {"schema_version": 1}))
        features = configuration.get("features", [])
        if not any(feature.get("id") == feature_id for feature in features):
            raise ValueError(f"Unknown model feature: {feature_id}")
        configuration["features"] = [feature for feature in features if feature.get("id") != feature_id]
        self.set_model_3d(key, configuration)

    def assign_material(self, key, material_key, region="body"):
        from temsim.part_materials import part_material_updates
        updates = part_material_updates(self.part(key), material_key, region=region)
        for path, value in updates.items():
            if tuple(path[:2]) != ("parts", key) or len(path) != 3:
                raise ValueError("Invalid material assignment path")
            self.part(key)[path[2]] = deepcopy(value)
        self._remember()

    def undo(self):
        if self.can_undo:
            self._history_index -= 1
            self.document = deepcopy(self._history[self._history_index])

    def redo(self):
        if self.can_redo:
            self._history_index += 1
            self.document = deepcopy(self._history[self._history_index])

    def revert(self):
        self.document = deepcopy(self._baseline)
        self._history = [deepcopy(self.document)]
        self._history_index = 0

    def updates(self):
        originals = {part["key"]: part for part in self._baseline["parts"]}
        updates = {}
        additions = []
        for part in self.document["parts"]:
            if part["key"] not in originals:
                additions.append(deepcopy(part))
                continue
            original = originals[part["key"]]
            for field, value in part.items():
                if field not in original or value != original[field]:
                    updates[("parts", part["key"], field)] = deepcopy(value)
            if "material_regions" in original and "material_regions" not in part:
                updates[("parts", part["key"], "material_regions")] = {}
            if "model_3d" in original and "model_3d" not in part:
                updates[("parts", part["key"], "model_3d")] = None
        current_keys = {part["key"] for part in self.document["parts"]}
        removed = tuple(key for key in originals if key not in current_keys)
        return PartChangeSet(updates, tuple(additions), removed, self._source_bytes)

    def reload(self):
        """Read an external source revision only after this draft is resolved."""
        if self.dirty:
            raise ValueError("Save a copy or Revert the current draft before reloading")
        candidate = type(self)(self.path)
        self._source_bytes = candidate._source_bytes
        self._baseline = candidate._baseline
        self.document = candidate.document
        self._history = candidate._history
        self._history_index = 0

    def validate(self):
        validate_component_graph(self.document)
        module_manifest.validate_document(self.document)

    def assert_source_current(self):
        if self.path.read_bytes() != self._source_bytes:
            raise ValueError("The source file changed outside this editor. Save a copy or reopen it before saving.")

    def save(self, *, project_save=None):
        self.assert_source_current()
        self.validate()
        updates = self.updates()
        if not updates:
            return
        staged = module_manifest.stage_manifest_text(
            self._source_bytes.decode("utf-8-sig"), updates
        )
        # Schema, Boolean and catalog preparation may take time. Recheck the
        # source before either persistence route can write the staged edit.
        self.assert_source_current()
        if project_save is not None:
            if project_save(self.path, updates) is False:
                raise ValueError("The project did not save the model draft")
        else:
            module_manifest._atomic_write_text(self.path, staged)
        self._accept_saved_file(self.path, expected=tomllib.loads(staged))

    def save_copy(self, path):
        destination = Path(path).resolve()
        if destination == self.path:
            raise ValueError("Use Save for the current file")
        self.validate()
        staged = module_manifest.stage_manifest_text(
            self._source_bytes.decode("utf-8-sig"), self.updates()
        )
        module_manifest._atomic_write_text(destination, staged)
        self._accept_saved_file(destination, expected=tomllib.loads(staged))

    def _accept_saved_file(self, path, *, expected):
        source = path.read_bytes()
        parsed = tomllib.loads(source.decode("utf-8-sig"))
        if parsed != expected:
            raise ValueError("The saved file does not match the draft. The draft is retained; inspect the source before saving again.")
        self.path = path
        self._source_bytes = source
        self._baseline = parsed
        self.document = deepcopy(self._baseline)
        self._history[self._history_index] = deepcopy(self.document)
