# Stable instrument navigation layout — 2026-09-26

Switching from Assembly to Direct Alignment could shrink the setup viewport
from 292 px to 68 px at a 745×887 test size. Only the control groups scrolled;
wrapping headings/status and the validation editor requested height from the
outer tab layout. Qt also propagated the largest stacked-page height through
the Mechanical tab's wrapping description.

All variable-height Direct Alignment content now shares a single resizable
scroll viewport. Cancel stays outside the viewport and remains directly
reachable. Control groups keep their separate rebuild host. Navigation tabs
no longer request height-for-width from the outer setup layout; ordinary
minimum sizes, internal wrapping, scrollbars and user splitter sizes remain.

The before/after GUI probe repeats Assembly → Direct Alignment → Optical →
Mechanical → Assembly → Direct Alignment. At 745×887 the setup viewport stays
292 px; at 420×650 it stays 213 px. The dock, tabbar and splitter also remain stable
with long status/validation text. A real MainWindow floating-dock regression
checks the 420×650 case without starting an electron or alignment calculation.
The existing parameter fields, mode gating, request/cancel signals and saved
layout identities are retained.

Local evidence: `tmp/alignment-layout-before.json`,
`tmp/alignment-layout-after.json`, `tmp/alignment-layout-after.png`, and
`tmp/check_alignment_layout.py`. Qt screenshots were inspected with the app
style and Segoe UI font. Generated evidence stays outside Git under `tmp/`.
This is a GUI change; restart the app to load it. Existing saved layouts do
not require a reset.

Validation: **34 targeted tests passed**, with zero failures/errors/skips.
This includes the new layout regressions and existing Direct Alignment and
workspace-layout checks. Syntax and `git diff --check` passed. This was not
a full-project test run.

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:PYTHONPATH='src;.'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:TEMSIM_CPU_THREADS='1'
.venv\Scripts\python.exe -m pytest -p pytestqt.plugin tests/test_alignment_panel_layout.py tests/test_gui_shell.py tests/test_workspace_layouts.py -k 'alignment or layout or component_selection_does_not_reset' -q --tb=short --junitxml=tmp/alignment-layout-regression.xml
```
