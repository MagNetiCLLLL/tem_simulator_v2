# Workspace layouts

Use **View > Layouts > Save layout as...** to create a named layout, for example
Teaching, Alignment or Sample analysis. Select its name from the same menu to
return to it. The checked name is the active layout.

Changes to the active layout are saved automatically after a short idle period
(600 ms), before changing layouts, and on normal exit. **Save current layout**
flushes changes immediately. Startup restores the last active layout. The
existing unnamed workspace is migrated into Default without resetting it.

Each layout retains:

- Main-window size, position and window state.
- Dock visibility, docking/tabbing, floating positions and floating sizes.
- Instrument editor, Live tuning, Ray Diagram, Sample, Sample Interactions 3D,
  Energy Filter, Scanning Image, Design Explorer and magnetic-validation splitters.
- Presentation tab selections, the magnetic/transverse panel switches and the
  Advanced bank expansion state.
- Separate Ray Diagram sizes for each magnetic/transverse on/off combination.

Hidden pages are restored when they are displayed, after Qt has allocated their
space. A hidden or collapsed panel must not overwrite its previously saved size.
Fixed-width editors retain their width while stretchable plots use the remaining
space. Qt minimum-size and monitor constraints still apply: a window cannot be
restored outside the available screen after changing monitors or display scale.

**View > Reset workspace layout** resets only the active layout. It does not
delete other layouts. Layout names are unique (case-insensitive), and an existing
name cannot be silently overwritten by Save layout as.

Layouts are presentation settings, not operating profiles: lens strengths,
sample data, physical parameters, plot zoom/camera pose and calculation caches
are not saved in them. Restoring a layout does not request optical, specimen,
image or spectrum calculations and does not discard results. Native file-picker
dialogs remain managed by Qt/the operating system.

## Implementation and checks

`gui/workspace_layouts.py` stores versioned snapshots in application QSettings
under `workspace_layouts/v1`. Layout identifiers are independent of display names.
Application splitters and tabs use unique, stable object names; unnamed or
ambiguous third-party widgets are not matched by fragile object indices.
New application panels should receive a stable name before the layout manager
is constructed. Ray panel changes save the old variant before hiding/showing
widgets, then restore the selected variant after layout. Presentation tab
signals are blocked during restoration to avoid initiating unrelated work.

`tests/test_workspace_layouts.py` covers named-layout isolation, restart,
previously hidden pages, ray-panel variants, autosave, menu selection, reset,
name validation and unchanged optical state/results. Offscreen tests use a fixed
viewport for splitter comparisons; native floating-window persistence is also
covered by `tests/test_live_tuning_dock.py`. These checks do not claim physical
monitor/DPI or native pointer-drag validation.
