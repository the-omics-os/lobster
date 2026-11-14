"""Dependency-free local GUI for already-saved Plotly figures."""


def launch_editor(store, ref):
    from .server import launch_editor as launch

    return launch(store, ref)
