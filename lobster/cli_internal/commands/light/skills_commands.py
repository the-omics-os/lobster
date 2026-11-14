"""``lobster skills``: list the skills shipped in the wheel and install them
into the skill folders of coding-agent harnesses (Claude Code, Codex, ...)."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from lobster.i18n import h

skills_app = typer.Typer(
    name="skills",
    help=h("List and install the agent skills shipped with Lobster."),
    no_args_is_help=True,
)

MARKER = ".lobster-skill.json"

# harness -> (home dir that proves it is installed, skill folder), both under ~.
# Paths as each harness documents them.
HARNESSES = {
    "claude-code": (".claude", ".claude/skills"),
    "codex": (".codex", ".codex/skills"),
    "cursor": (".cursor", ".cursor/skills"),
    "os": (".os", ".os/agent/skills"),
    "pi": (".pi", ".pi/agent/skills"),
}


def skills_source() -> Path:
    """``lobster/skills``: package data in the wheel, the canonical tree in a checkout."""
    return Path(__file__).resolve().parents[3] / "skills"


def list_skills() -> List[Dict[str, str]]:
    source = skills_source()
    if not source.is_dir():
        return []
    return [
        {"name": path.name, "path": str(path)}
        for path in sorted(source.iterdir())
        if (path / "SKILL.md").is_file()
    ]


def _targets(harnesses: List[str], target: Optional[Path]) -> Dict[str, Path]:
    if target is not None:
        return {"target": target.expanduser()}
    home = Path.home()
    names = harnesses or [n for n, (d, _) in HARNESSES.items() if (home / d).is_dir()]
    return {name: home / HARNESSES[name][1] for name in names}


def _installed_by_lobster(dest: Path) -> bool:
    """Only a real directory with our JSON marker may be replaced (removed)."""
    if dest.is_symlink() or not dest.is_dir():
        return False
    try:
        marker = json.loads((dest / MARKER).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(marker, dict) and marker.get("installed_by") == "lobster-ai"


def plan_install(harnesses: List[str], target: Optional[Path]) -> List[Dict[str, Any]]:
    actions = []
    for harness, folder in _targets(harnesses, target).items():
        for skill in list_skills():
            dest = folder / skill["name"]
            action = {
                "harness": harness,
                "skill": skill["name"],
                "source": skill["path"],
                "destination": str(dest),
                "action": "install",
            }
            if dest.is_symlink() or dest.exists():
                ours = _installed_by_lobster(dest)
                action["action"] = "replace" if ours else "skip"
                if not ours:
                    action["reason"] = "existing skill was not installed by lobster"
            actions.append(action)
    return actions


def apply_install(actions: List[Dict[str, Any]]) -> None:
    from lobster.version import __version__

    for action in actions:
        if action["action"] == "skip":
            continue
        dest = Path(action["destination"])
        if action["action"] == "replace":
            # Check again: the folder may have changed since the plan.
            # shortcut: a change between this check and rmtree is not caught;
            # an atomic rename-aside would close it if that ever matters.
            if not _installed_by_lobster(dest):
                action["action"] = "skip"
                action["reason"] = "destination changed after the check; left as is"
                continue
            shutil.rmtree(dest)
        shutil.copytree(
            action["source"],
            dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        marker = {"installed_by": "lobster-ai", "version": __version__}
        (dest / MARKER).write_text(json.dumps(marker, indent=2) + "\n")


@skills_app.command(name="list")
def skills_list(
    as_json: bool = typer.Option(
        False, "--json", help=h("Print the shipped skills as JSON.")
    ),
):
    skills = list_skills()
    if as_json:
        payload = {
            "schema_version": "lobster.skills/1",
            "source": str(skills_source()),
            "skills": skills,
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    for skill in skills:
        typer.echo(f"{skill['name']:<24} {skill['path']}")


skills_list.__doc__ = h("List the agent skills shipped with Lobster.")


@skills_app.command(name="install")
def skills_install(
    harness: Optional[List[str]] = typer.Option(
        None,
        "--harness",
        help=h(
            "Harness to install into (claude-code, codex, cursor, os, pi); "
            "repeatable. Default: every harness found in your home directory."
        ),
    ),
    target: Optional[Path] = typer.Option(
        None, "--target", help=h("Install into this skill folder instead.")
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help=h("Show what would be copied; write nothing.")
    ),
    as_json: bool = typer.Option(
        False, "--json", help=h("Print the lobster.skills-install/1 plan as JSON.")
    ),
):
    unknown = sorted(set(harness or []) - set(HARNESSES))
    if unknown:
        typer.echo(
            h("Unknown harness: {names}. Choose from: {choices}").format(
                names=", ".join(unknown), choices=", ".join(HARNESSES)
            ),
            err=True,
        )
        raise typer.Exit(2)
    actions = plan_install(harness or [], target)
    if not dry_run:
        apply_install(actions)
    if as_json:
        payload = {
            "schema_version": "lobster.skills-install/1",
            "dry_run": dry_run,
            "actions": actions,
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for a in actions:
            typer.echo(f"{a['action']:<8} {a['source']} -> {a['destination']}")
            if a["action"] == "skip":
                typer.echo(
                    h("Skipped {path}: it was not installed by lobster.").format(
                        path=a["destination"]
                    ),
                    err=True,
                )
    if not actions:
        typer.echo(
            h("No harness skill folder found; use --harness or --target."), err=True
        )
        raise typer.Exit(1)


skills_install.__doc__ = h(
    "Copy the shipped skills into coding-agent skill folders (Claude Code, Codex, ...)."
)
