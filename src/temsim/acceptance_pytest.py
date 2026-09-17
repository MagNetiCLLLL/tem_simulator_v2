"""Exact collection and three-phase outcomes for the bounded acceptance CLI."""
import json
import os
from pathlib import Path

_receipt = {"collected": [], "cases": {}, "deselected": [], "complete": False}


def pytest_collection_finish(session):
    _receipt["collected"] = [item.nodeid for item in session.items]


def pytest_deselected(items):
    _receipt["deselected"].extend(item.nodeid for item in items)


def pytest_runtest_logreport(report):
    _receipt["cases"].setdefault(report.nodeid, {})[report.when] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    _receipt["complete"] = True
    _receipt["exitstatus"] = int(exitstatus)
    Path(os.environ["TEMSIM_ACCEPTANCE_RECEIPT"]).write_text(json.dumps(_receipt, indent=2), encoding="utf-8")
