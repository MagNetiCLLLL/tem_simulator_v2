"""Optional material-table edits and exact transaction snapshots."""

from copy import deepcopy
from pathlib import Path
import shutil
import tomllib

import pytest
import tomli_w

from temsim import module_manifest
from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.part_materials import material_catalog
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = "project_and_recording_system/EnergyFilter.toml"
KEY = "intermediate_lens_excitation_coil"


@pytest.mark.parametrize("layout", ["inline", "dotted", "escaped_key", "expanded"])
@pytest.mark.parametrize("empty", [False, True])
def test_material_table_replacement_preserves_unrelated_tables_and_comments(layout, empty):
    materials = {row["material_key"]: row for row in material_catalog()}
    previous = {"body": materials["copper"]}
    source = (INSTRUMENT_CONFIG_ROOT / MODULE).read_bytes().decode("utf-8")
    lines = source.splitlines(keepends=True)
    start, end = module_manifest._part_span(lines, KEY)
    before_table = '[parts.custom_before]\ntext = "unchanged # literal"\nmaterial_regions = "unrelated metadata"\n'
    after_table = '[parts.custom_after]\ntext = "also unchanged" # neighbour comment\n'
    if layout == "expanded":
        table = tomli_w.dumps({"parts": [{"material_regions": previous}]}).split("\n", 1)[1]
        table = table.replace("[parts.material_regions.body]", '[parts."material_regions".body] # material table comment')
        table = table.replace('material_key = "copper"', 'material_key = "copper" # assignment comment')
        lines[end:end] = [before_table, "# retained material rationale\n", table, after_table]
    else:
        field = "material_regions" if layout == "inline" else '"material_regions".body'
        if layout == "escaped_key":
            field = r'"material\u005fregions".body'
        value = previous if layout == "inline" else previous["body"]
        assignment = f"{field} = {module_manifest._format_toml_value(value)} # assignment comment\n"
        lines[end:end] = [before_table, after_table]
        lines[start:start] = ["# retained material rationale\n", assignment]
    source = "".join(lines)
    expected = deepcopy(tomllib.loads(source))
    module_manifest.validate_document(expected)
    replacement = {} if empty else {"body": materials["aluminum"]}
    next(part for part in expected["parts"] if part["key"] == KEY)["material_regions"] = replacement
    staged = module_manifest.stage_manifest_text(source, {("parts", KEY, "material_regions"): replacement})
    assert tomllib.loads(staged) == expected
    assert before_table in staged
    assert after_table in staged
    assert "# retained material rationale" in staged
    assert "# assignment comment" in staged
    if layout == "expanded":
        assert "# material table comment" in staged


def test_table_comment_retention_does_not_extract_hashes_inside_strings():
    lines = [
        'label = "value # literal" # keep double\n',
        "literal = '# literal' # keep single\r\n",
        'multiline = """value # literal\n',
        '# still inside the string\n',
        'ends with a quote"""" # keep multiline\n',
        'array = ["x", # keep array\n',
        '"y"] # keep array end\n',
        'escaped = "quoted \\" # still literal" # keep escaped\n',
        "multiliteral = '''# literal\n",
        "''' # keep multiline literal\n",
    ]
    tomllib.loads("".join(lines))
    comments = module_manifest._retained_toml_comments(lines, "\n")
    assert [line.strip() for line in comments] == [
        "# keep double", "# keep single", "# keep multiline", "# keep array",
        "# keep array end", "# keep escaped", "# keep multiline literal",
    ]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "mixed"])
def test_transaction_snapshot_and_explicit_restore_preserve_original_bytes(tmp_path, newline):
    path = tmp_path / "module.toml"
    source = (INSTRUMENT_CONFIG_ROOT / MODULE).read_bytes().decode("utf-8").replace("\r\n", "\n")
    if newline == "mixed":
        source = source.replace("\n", "\r\n", 3)
    else:
        source = source.replace("\n", newline)
    original = source.encode("utf-8")
    path.write_bytes(original)
    snapshot = module_manifest.update_manifest_values(
        {path.name: {("parts", KEY, "name"): "Edited coil"}}, root=tmp_path)
    assert path.read_bytes() != original
    assert snapshot[path.name].encode("utf-8") == original
    module_manifest.restore_manifest_texts(snapshot, root=tmp_path)
    assert path.read_bytes() == original


def test_catalog_failure_rolls_back_crlf_source_exactly(tmp_path, monkeypatch):
    path = tmp_path / MODULE
    path.parent.mkdir(parents=True)
    original = (INSTRUMENT_CONFIG_ROOT / MODULE).read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
    path.write_bytes(original)
    editor = ManifestEditor(tmp_path)
    def fail():
        raise ValueError("Synthetic catalog failure")
    monkeypatch.setattr(editor, "validate_catalog", fail)
    with pytest.raises(ValueError, match="Synthetic catalog failure"):
        editor.save(ManifestTarget(MODULE, KEY), {("parts", KEY, "name"): "Edited coil"}, None)
    assert path.read_bytes() == original


def test_later_atomic_write_failure_restores_prior_crlf_module(tmp_path, monkeypatch):
    first = tmp_path / "first.toml"
    original = (INSTRUMENT_CONFIG_ROOT / MODULE).read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
    first.write_bytes(original)
    second = tmp_path / "second.toml"
    shutil.copyfile(first, second)
    write = module_manifest._atomic_write_text
    def fail_second(path, text):
        if Path(path) == second:
            raise OSError("Synthetic write failure")
        write(path, text)
    monkeypatch.setattr(module_manifest, "_atomic_write_text", fail_second)
    updates = {("parts", KEY, "name"): "Edited coil"}
    with pytest.raises(OSError, match="Synthetic write failure"):
        module_manifest.update_manifest_values({first.name: updates, second.name: updates}, root=tmp_path)
    assert first.read_bytes() == original
    assert second.read_bytes() == original
