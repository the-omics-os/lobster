"""Lazy namespace discovery and declared recipe metadata for drawing functions."""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from functools import wraps

MAX_TEMPLATE_ARGS_BYTES = 16 * 1024


def figure_template(name: str, *, version: str = "1", data_args=("data",)):
    """Record resolved, data-free settings; this metadata is declared, not proof."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def draw(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            settings = {
                key: value
                for key, value in bound.arguments.items()
                if key not in data_args
            }
            try:
                payload = json.dumps(settings, allow_nan=False, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Template settings must be finite JSON values, without data"
                ) from exc
            if len(payload.encode("utf-8")) > MAX_TEMPLATE_ARGS_BYTES:
                raise ValueError(
                    "Template settings exceed the 16 KiB limit; nothing was drawn"
                )
            figure = function(*args, **kwargs)
            from lobster.figure_templates._style import apply_house_style

            apply_house_style(figure)
            meta = dict(figure.layout.meta or {})
            meta["lobster_template"] = {
                "name": name,
                "version": version,
                "args": json.loads(payload),
            }
            figure.update_layout(meta=meta)
            # Record only template parameters whose column meaning is explicit.
            # Matrix/single-cell templates do not claim tabular column mappings.
            roles, transformations = [], []
            selection = {}
            if name in {
                "generic.scatter",
                "generic.line",
                "generic.bar",
                "generic.box",
                "generic.histogram",
            }:
                fields = {
                    "x": "x",
                    "y": "y",
                    "color": "color",
                    "size": "size",
                    "id_col": "id",
                }
                roles = [
                    {"column": settings[key], "role": role}
                    for key, role in fields.items()
                    if isinstance(settings.get(key), str)
                ]
                # Automatic identity selection is declared by the drawing code,
                # not inferred later from values or axis labels.
                if name == "generic.scatter" and not isinstance(
                    settings.get("id_col"), str
                ):
                    id_columns = {
                        t.meta["lobster_ids"]["id_column"]
                        for t in figure.data
                        if isinstance(t.meta, dict)
                        and t.meta.get("lobster_ids", {}).get("id_column")
                    }
                    if len(id_columns) == 1:
                        roles.append({"column": id_columns.pop(), "role": "id"})
                if name in {"generic.histogram", "generic.box"}:
                    transformations = [
                        {
                            "operation": name,
                            "description": "Display summarizes the supplied table.",
                        }
                    ]
            elif name == "proteomics.volcano":
                roles = [
                    {"column": settings["effect_field"], "role": "x"},
                    {
                        "column": settings["p_field"],
                        "role": "y",
                        "transform": "−log10(max(value, p_floor))",
                    },
                    {"column": settings["entity_field"], "role": "id"},
                    {"column": settings["comparison_field"], "role": "filter"},
                ]
                if settings["label_top"]:
                    roles.append({"column": settings["entity_field"], "role": "label"})
                transformations = [
                    {
                        "operation": "volcano",
                        "description": "Y uses −log10(max(p, p_floor)); trace colors use effect/p-value cutoffs.",
                        "p_floor": settings["p_floor"],
                        "fdr": settings["fdr"],
                        "log2fc_min": settings["log2fc_min"],
                        "boundary": settings["boundary"],
                    }
                ]
                selection = {
                    "method": "comparison filter",
                    "column": settings["comparison_field"],
                    "equals": settings["comparison"],
                }
            if roles:
                from lobster.core.figures.data_view import record_column_roles

                record_column_roles(
                    figure, roles, transformations=transformations, selection=selection
                )
            return figure

        draw.template_name = name
        draw.template_version = version
        return draw

    return decorate


def _catalog():
    # PEP 420 makes providers from installed domain packages discoverable without
    # hardcoded package lists or changes to the core/agent component registries.
    namespace = importlib.import_module("lobster.figure_template_providers")
    result = {}
    for module_info in sorted(
        pkgutil.iter_modules(namespace.__path__), key=lambda item: item.name
    ):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{namespace.__name__}.{module_info.name}")
        for item in getattr(module, "TEMPLATE_MANIFEST", ()):
            descriptor = dict(item)
            name = descriptor["name"]
            if name in result:
                raise ValueError(f"Duplicate figure template: {name}")
            function = getattr(module, descriptor.pop("callable"))
            if (
                getattr(function, "template_name", None) != name
                or getattr(function, "template_version", None) != descriptor["version"]
            ):
                raise ValueError(f"Template manifest/version mismatch: {name}")
            descriptor["arguments"] = str(inspect.signature(function))
            result[name] = (descriptor, function)
    return result


def list_templates():
    """Return names, versions, one-line purposes and callable signatures on demand."""
    catalog = _catalog()
    return [catalog[name][0] for name in sorted(catalog)]


def get_template(name: str):
    """Get a normal drawing callable, optionally pinned as 'proteomics.volcano@1'."""
    base, separator, version = name.partition("@")
    catalog = _catalog()
    if base not in catalog:
        raise KeyError(f"Unknown or unavailable template: {name}")
    descriptor, function = catalog[base]
    if separator and version != descriptor["version"]:
        raise KeyError(f"Unavailable template version: {name}")
    return function


def show_template_source(name: str):
    """Return the readable provider source, including helpers needed to adapt it."""
    function = inspect.unwrap(get_template(name))
    return inspect.getsource(inspect.getmodule(function))
