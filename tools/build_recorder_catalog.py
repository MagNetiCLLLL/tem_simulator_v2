"""Build the read-only recorder catalog from a user-supplied AutoScript wheel.

No SDK code is imported or executed. Output is JSON on stdout. The proprietary
wheel stays outside the repository; only API paths, types and unit notes remain.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile


ROOTS = ("optics", "source", "specimen", "detectors", "vacuum", "service")
EXCLUDE = {
    "service.simulator": "Simulator control is not instrument metrology.",
    "service.internal": "Undocumented internal interface.",
    "service.offline": "Offline service is not a real microscope.",
    "service.toolkit": "Toolkit service is not an optical readback.",
    "optics.aperture_mechanisms.condenser_1": "Read via enumerated mechanisms.",
    "optics.aperture_mechanisms.condenser_2": "Read via enumerated mechanisms.",
    "optics.aperture_mechanisms.condenser_3": "Read via enumerated mechanisms.",
    "optics.aperture_mechanisms.objective": "Read via enumerated mechanisms.",
    "optics.aperture_mechanisms.selected_area": "Read via enumerated mechanisms.",
}


def getters(node):
    return [f for f in node.body if isinstance(f, ast.FunctionDef)
            and any(isinstance(d, ast.Name) and d.id == "property" for d in f.decorator_list)
            and not f.name.startswith("_")]


def build(path: Path):
    classes = {}
    with zipfile.ZipFile(path) as archive:
        metadata = archive.read("autoscript_tem_microscope_client-1.18.0.dist-info/METADATA").decode()
        if "Version: 1.18.0" not in metadata.splitlines():
            raise ValueError("Only the audited AutoScript 1.18.0 client is supported.")
        for name in archive.namelist():
            if not name.endswith(".py"):
                continue
            module = name[:-3].replace("/", ".")
            tree = ast.parse(archive.read(name))
            imports = {}
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    target = (("." * node.level) + (node.module or ""))
                    if node.level:
                        target = importlib.util.resolve_name(target, module.rpartition(".")[0])
                    for alias in node.names:
                        imports[alias.asname or alias.name] = (target, alias.name)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes[(module, node.name)] = (node, imports)

    paths = []
    def visit(key, prefix, depth=0):
        if depth > 12:
            raise ValueError(f"Unexpected proxy recursion: {prefix}")
        node, imports = classes[key]
        for prop in getters(node):
            full = f"{prefix}.{prop.name}" if prefix else prop.name
            if not prefix and prop.name not in ROOTS:
                continue
            if full in EXCLUDE:
                continue
            annotation = prop.returns
            typename = (annotation.value if isinstance(annotation, ast.Constant)
                        else ast.unparse(annotation) if annotation else "unknown")
            target = imports.get(typename)
            if target in classes and ".tem_microscope." in target[0]:
                visit(target, full, depth + 1)
            else:
                # Dynamic proxy properties require enumerated identities below.
                if target and target[0].endswith("_dynamic_object_proxies"):
                    continue
                doc = " ".join((ast.get_docstring(prop) or "").split())
                paths.append({"path": full, "type": typename, "description": doc})

    visit(("autoscript_tem_microscope_client.tem_microscope_client", "TemMicroscopeClient"), "")
    structures, dynamic = {}, {}
    for (module, name), (node, _) in classes.items():
        fields = [{"name": f.name,
                   "type": (f.returns.value if isinstance(f.returns, ast.Constant)
                            else ast.unparse(f.returns) if f.returns else "unknown"),
                   "description": " ".join((ast.get_docstring(f) or "").split())}
                  for f in getters(node)]
        if module.endswith("_dynamic_object_proxies"):
            dynamic[name] = fields
        elif "structures" in module and fields:
            structures[name] = fields
    for field in dynamic["Lens"]:
        if field["name"] == "value_raw":
            field["description"] = ("Raw output in optical units. Scalar float / wire DOUBLE in SDK 1.18.0; "
                                    "the vendor getter's 2D-vector docstring is inconsistent with its signature and wire type.")
    return {"sdk_version": "1.18.0", "wheel_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "properties": paths, "dynamic_properties": dynamic, "structures": structures,
            "exclusions": {**EXCLUDE,
                "acquisition / auto_functions": "Actions, not passive machine parameters.",
                "analysis": "Acquisition streams and jobs, not passive machine parameters.",
                "state": "Saved state / APT experiments are not live optical readbacks.",
                "service.autoscript.server.configuration.get_value": "Requires a caller-supplied key; SDK provides no key enumerator.",
                "methods": "Only explicitly implemented enumerators and read methods are invoked."}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.wheel), indent=2, ensure_ascii=True))
