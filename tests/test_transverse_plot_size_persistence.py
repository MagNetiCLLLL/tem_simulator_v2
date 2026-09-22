"""Fixed transverse picture sizes follow named layouts without physics work."""
from copy import deepcopy

from test_workspace_layouts import windows  # noqa: F401


SMALL = {"source": [360, 210], "plane": [440, 330], "legend": [200, 200]}
LARGE = {"source": [520, 280], "plane": [560, 420], "legend": [240, 260]}


def test_named_layouts_restore_distinct_picture_sizes_without_changing_inputs(windows):
    make, _ = windows
    window = make()
    manager, view = window.workspace_layouts, window.workspace.transverse_beam
    before_state = deepcopy(window.state.to_dict())
    revision = window._physical_revision
    view.set_plot_size_state(SMALL)
    custom = manager.save_as("Larger beam pictures")
    view.set_plot_size_state(LARGE)
    manager.save_current()
    manager.select("default")
    assert view.plot_size_state() == SMALL
    manager.select(custom)
    assert view.plot_size_state() == LARGE
    assert window.state.to_dict() == before_state
    assert window._physical_revision == revision
    assert not window.preview_timer.isActive()


def test_picture_size_change_autosaves_to_active_layout(windows, qtbot):
    make, settings = windows
    window = make()
    manager, view = window.workspace_layouts, window.workspace.transverse_beam
    manager.save_timer.setInterval(20)
    view.set_plot_size_state(SMALL)

    def saved_sizes():
        saved = settings.value(manager._key(manager.active_id, "data"), {})
        return saved.get("transverse_plot_sizes") if isinstance(saved, dict) else None

    qtbot.waitUntil(lambda: saved_sizes() == SMALL)
    assert not window.preview_timer.isActive()


def test_restart_restores_picture_sizes_while_transverse_panel_is_hidden(windows):
    make, settings = windows
    window = make()
    manager = window.workspace_layouts
    window.workspace.transverse_beam.set_plot_size_state(LARGE)
    window.workspace.transverse_beam_toggle.setChecked(False)
    saved_id = manager.save_as("Hidden beam pictures")
    window.close()
    assert settings.value(manager._key(saved_id, "data"))["transverse_plot_sizes"] == LARGE

    restored = make()
    assert restored.workspace_layouts.active_id == saved_id
    assert not restored.workspace.transverse_beam_toggle.isChecked()
    assert restored.workspace.transverse_beam.plot_size_state() == LARGE
    restored.workspace.transverse_beam_toggle.setChecked(True)
    assert restored.workspace.transverse_beam.plot_size_state() == LARGE
    assert not restored.preview_timer.isActive()


def test_reset_picture_sizes_only_changes_selected_layout(windows):
    make, _ = windows
    window = make()
    manager, view = window.workspace_layouts, window.workspace.transverse_beam
    defaults = view.plot_size_state()
    view.set_plot_size_state(SMALL)
    custom = manager.save_as("Reset beam pictures")
    view.set_plot_size_state(LARGE)
    manager.reset_current()
    assert view.plot_size_state() == defaults
    manager.select("default")
    assert view.plot_size_state() == SMALL
    manager.select(custom)
    assert view.plot_size_state() == defaults


def test_legacy_layout_uses_default_picture_sizes_without_save_during_restore(windows):
    make, settings = windows
    window = make()
    manager, view = window.workspace_layouts, window.workspace.transverse_beam
    defaults = view.plot_size_state()
    legacy = deepcopy(manager._snapshot())
    legacy.pop("transverse_plot_sizes")
    settings.setValue(manager._key("legacy", "name"), "Legacy")
    settings.setValue(manager._key("legacy", "data"), legacy)
    view.set_plot_size_state(LARGE)
    manager.select("legacy")
    assert view.plot_size_state() == defaults
    assert not manager.save_timer.isActive()
    # Applying a historical snapshot does not rewrite it on construction or
    # trigger a calculation. A later real user change is saved normally.
    assert "transverse_plot_sizes" not in settings.value(manager._key("legacy", "data"))
    assert not window.preview_timer.isActive()
