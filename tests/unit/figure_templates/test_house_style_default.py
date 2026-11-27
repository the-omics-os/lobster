"""A DataManagerV2 session makes omics_os the default for every Plotly figure."""

import sys

import plotly.express as px
import plotly.io as pio
import pytest

from lobster.core.runtime.data_manager import DataManagerV2
from lobster.figure_templates._style import (
    HOUSE_COLORS,
    HOUSE_STYLE_NAME,
    SEQUENTIAL,
)


@pytest.fixture
def restore_plotly_templates():
    """Other tests must not inherit the house style from this one."""
    previous_default = pio.templates.default
    previous_template = (
        pio.templates[HOUSE_STYLE_NAME] if HOUSE_STYLE_NAME in pio.templates else None
    )
    try:
        pio.templates.default = "plotly"
        if previous_template is not None:
            del pio.templates[HOUSE_STYLE_NAME]
        yield
    finally:
        if HOUSE_STYLE_NAME in pio.templates:
            del pio.templates[HOUSE_STYLE_NAME]
        if previous_template is not None:
            pio.templates[HOUSE_STYLE_NAME] = previous_template
        pio.templates.default = previous_default


def test_session_figures_embed_house_style(tmp_path, restore_plotly_templates):
    DataManagerV2(workspace_path=tmp_path)
    assert pio.templates.default == HOUSE_STYLE_NAME

    figure = px.scatter(
        x=[0, 1, 2, 3, 4], y=[1, 0, 1, 0, 1], color=["a", "b", "c", "a", "b"]
    )
    assert [trace.marker.color for trace in figure.data] == HOUSE_COLORS[:3]
    colorscale = figure.layout.template.layout.colorscale.sequential
    assert [list(stop) for stop in colorscale] == SEQUENTIAL


def test_session_works_without_templates(
    tmp_path, monkeypatch, restore_plotly_templates
):
    # None in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "lobster.figure_templates", None)
    manager = DataManagerV2(workspace_path=tmp_path)
    assert manager.workspace_path.exists()
    assert pio.templates.default == "plotly"


def test_session_keeps_a_default_someone_chose(tmp_path, restore_plotly_templates):
    pio.templates.default = "plotly_white"
    DataManagerV2(workspace_path=tmp_path)
    assert pio.templates.default == "plotly_white"
    assert HOUSE_STYLE_NAME in pio.templates  # still registered for templates
