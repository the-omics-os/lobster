"""Chinese status, slash help and non-interactive init summary.

Machine output (JSON) must stay identical in every UI language.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess  # nosec B404 # test runs the CLI as a child process
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from lobster.i18n import set_lang, use_lang

ROOT = Path(__file__).resolve().parents[3]
HEAVY = ROOT / "lobster" / "cli_internal" / "commands" / "heavy"
KEY = "sk-ant-synthetic-test-key"  # nosec B105 - synthetic value


@pytest.fixture(autouse=True)
def reset_locale():
    set_lang("en")
    yield
    set_lang("en")


def _modules():
    # Other tests purge lobster.cli* from sys.modules: resolve at call time.
    return (
        importlib.import_module("lobster.cli_internal.commands.heavy.display_helpers"),
        importlib.import_module("lobster.cli_internal.commands.heavy.slash_commands"),
    )


@pytest.fixture
def status_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "omics-os")

    import lobster.config.agent_registry as agent_registry
    import lobster.config.subscription_tiers as subscription_tiers
    import lobster.core.license_manager as license_manager
    import lobster.core.plugin_loader as plugin_loader

    monkeypatch.setattr(
        license_manager,
        "get_entitlement_status",
        lambda: {
            "tier": "free",
            "tier_display": "Free",
            "source": "default",
            "features": [f"feature_{i}" for i in range(8)],
        },
    )
    monkeypatch.setattr(
        plugin_loader,
        "get_installed_packages",
        lambda: {"lobster-ai": "1.0.0", "omics": "dev", "gone": "missing"},
    )
    monkeypatch.setattr(
        agent_registry,
        "get_worker_agents",
        lambda: {f"agent_{i}": object() for i in range(9)},
    )
    monkeypatch.setattr(subscription_tiers, "is_agent_available", lambda n, t: True)
    return tmp_path


def _texts(blocks):
    out = []
    for block in blocks:
        data = block.data
        for key in ("title", "body", "message"):
            if data.get(key):
                out.append(str(data[key]))
        for item in data.get("items", []):
            out.append(str(item))
        for col in data.get("columns", []):
            out.append(str(col.get("name", "")))
        for key in ("key_label", "value_label"):
            if data.get(key):
                out.append(str(data[key]))
        for row in data.get("rows", []):
            out.extend(str(cell) for cell in row)
    return out


def test_status_blocks_are_chinese_and_keep_dynamic_data(status_env):
    display_helpers, _ = _modules()
    set_lang("zh")
    blocks = display_helpers.build_status_blocks()
    text = _texts(blocks)
    for expected in (
        "Lobster 状态",
        "初始化",
        "已配置",
        "模型服务",
        "配置来源",
        "账户访问",
        "未检查（仅检查配置）",
        "字段",
        "已安装的软件包",
        "软件包",
        "缺失",
        "开发版",
        "已安装",
        "可选功能",
        "已启用功能",
        "可用智能体（9）",
    ):
        assert expected in text, expected
    # Dynamic data stays as-is.
    assert "omics-os" in text
    assert "agent_0" in text
    assert "lobster-ai" in text
    assert "Free" in text
    # No leftover English fixed labels.
    for english in ("Lobster Status", "Configuration Source", "Account Access"):
        assert english not in text


def test_status_blocks_english_is_unchanged(status_env):
    display_helpers, _ = _modules()
    text = _texts(display_helpers.build_status_blocks())
    for expected in (
        "Lobster Status",
        "Initialization",
        "Configured",
        "Configuration Source",
        "Not checked (configuration only)",
        "Field",
        "Available Agents (9)",
        "Enabled Features",
    ):
        assert expected in text, expected


def test_compact_status_hints_are_localized_with_counts(status_env):
    display_helpers, _ = _modules()
    set_lang("zh")
    text = _texts(display_helpers.build_status_blocks(compact=True))
    assert "运行概况" in text
    assert "运行 `lobster status` 查看完整的软件包和智能体列表。" in text
    set_lang("en")
    english = _texts(display_helpers.build_status_blocks(compact=True))
    assert "Runtime Summary" in english


def test_status_more_hints_in_both_languages(status_env, monkeypatch):
    display_helpers, _ = _modules()
    # The "... and N more" hints only exist for compact lists; exercise the keys.
    from lobster.i18n import t

    set_lang("zh")
    assert t("cli.status.more_agents", n=3) == "……还有 3 个可用智能体"
    assert t("cli.status.more_features", n=2) == "……还有 2 个已启用功能"
    assert t("cli.status.license_expires", days=5) == "许可证将在 5 天后到期"
    set_lang("en")
    assert t("cli.status.more_agents", n=3) == "... and 3 more available agents"
    assert t("cli.status.license_expires", days=5) == "License expires in 5 days"


def test_help_blocks_translate_prose_and_keep_commands(status_env):
    _, slash_commands = _modules()
    english = slash_commands._build_help_blocks()
    set_lang("zh")
    chinese = slash_commands._build_help_blocks()
    zh_text = _texts(chinese)
    for expected in (
        "快速开始",
        "常用命令",
        "进阶命令",
        "管理与界面命令",
        "命令",
        "说明",
        "退出",
        "当前会话状态",
    ):
        assert expected in zh_text, expected
    en_tables = [b for b in english if b.kind == "table"]
    zh_tables = [b for b in chinese if b.kind == "table"]
    assert len(en_tables) == len(zh_tables) == 3
    for en_table, zh_table in zip(en_tables, zh_tables):
        assert [r[0] for r in en_table.data["rows"]] == [
            r[0] for r in zh_table.data["rows"]
        ]
        assert all(r[0].startswith("/") for r in zh_table.data["rows"])
        # Every description cell has a Chinese translation.
        for en_row, zh_row in zip(en_table.data["rows"], zh_table.data["rows"]):
            assert en_row[1] != zh_row[1], en_row

    set_lang("zh")
    protocol = slash_commands._help_table_columns(protocol_friendly=True)
    assert [c["name"] for c in protocol] == ["命令", "说明"]


def test_query_help_is_translated(status_env):
    _, slash_commands = _modules()
    set_lang("zh")
    text = _texts(slash_commands._build_query_help_blocks())
    assert "查询模式帮助" in text
    assert "查询模式可用命令" in text
    assert "/pipeline export" in text


def _json_dict(slash_commands, command, lang):
    from lobster.cli_internal.commands.output_adapter import JsonOutputAdapter

    set_lang(lang)
    output = JsonOutputAdapter()
    client = SimpleNamespace(workspace_path=None, provider_override=None)
    slash_commands._execute_command(command, client, command, output)
    return output.to_dict()


def _key_paths(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{prefix}/{key}"
            yield from _key_paths(item, f"{prefix}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _key_paths(item, f"{prefix}[{index}]")


def test_status_json_is_identical_in_english_and_chinese(status_env):
    _, slash_commands = _modules()
    english = _json_dict(slash_commands, "/status", "en")
    chinese = _json_dict(slash_commands, "/status", "zh")
    assert english["tables"], "status JSON should contain tables"
    assert list(_key_paths(english)) == list(_key_paths(chinese))
    assert json.dumps(english, sort_keys=True) == json.dumps(chinese, sort_keys=True)
    assert english["tables"][0]["columns"] == ["Field", "Value"]
    assert any(row[0] == "Initialization" for row in english["tables"][0]["rows"])


def test_help_json_is_identical_in_english_and_chinese(status_env):
    _, slash_commands = _modules()
    english = _json_dict(slash_commands, "/help", "en")
    chinese = _json_dict(slash_commands, "/help", "zh")
    assert english["tables"][0]["columns"] == ["Command", "Description"]
    assert json.dumps(english, sort_keys=True) == json.dumps(chinese, sort_keys=True)


def test_use_lang_restores_language():
    set_lang("zh")
    from lobster.i18n import current_lang, h

    with use_lang("en"):
        assert h("Clear screen") == "Clear screen"
    assert current_lang() == "zh"
    assert h("Clear screen") == "清屏"


def _h_constants(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "h"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            yield node.args[0].value


def test_status_and_help_phrases_exist_and_are_used():
    phrases = json.loads(
        (ROOT / "lobster" / "i18n" / "phrases.zh.json").read_text(encoding="utf-8")
    )
    used = set()
    for name in ("display_helpers.py", "slash_commands.py"):
        used.update(_h_constants(HEAVY / name))
    assert used
    assert sorted(used - set(phrases)) == []


@pytest.fixture
def init_env(monkeypatch, tmp_path):
    from lobster.config import credentials, global_config

    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", tmp_path / "creds")
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", tmp_path / "creds" / "credentials.json"
    )
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: False)
    for name in (
        "OMICS_OS_API_KEY",
        "OMICS_OS_ENDPOINT",
        "LOBSTER_LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "LOBSTER_WORKSPACE",
    ):
        monkeypatch.delenv(name, raising=False)
    return proj


def _run_init(monkeypatch, lang, extra=()):
    init_commands = importlib.import_module(
        "lobster.cli_internal.commands.heavy.init_commands"
    )
    console = Console(width=400, record=True, force_terminal=False)
    monkeypatch.setattr(init_commands, "console", console)
    # Importing lobster.cli resolves the language, so import before set_lang.
    cli_module = importlib.import_module("lobster.cli")
    set_lang(lang)
    result = CliRunner().invoke(
        cli_module.app,
        [
            "init",
            "--non-interactive",
            "--skip-extras",
            "--skip-ssl-test",
            "--anthropic-key",
            KEY,
            *extra,
        ],
    )
    return result, console.export_text()


def test_non_interactive_init_summary_is_chinese(monkeypatch, init_env):
    result, text = _run_init(monkeypatch, "zh", ("--model", "claude-test-model"))
    assert result.exit_code == 0, result.output
    assert "设置完成" in text
    assert "模型服务：anthropic (claude-test-model)" in text
    assert "工作区：" in text
    assert f"凭据文件：{init_env / '.env'}" in text
    assert "下一步：运行 lobster chat" in text
    assert "Configuration saved" not in text
    assert KEY not in text


def test_non_interactive_init_summary_is_english(monkeypatch, init_env):
    result, text = _run_init(monkeypatch, "en")
    assert result.exit_code == 0, result.output
    assert "Setup complete" in text
    assert "Provider: anthropic" in text
    assert f"Credentials: {init_env / '.env'}" in text
    assert "Next: run lobster chat" in text
    assert "Configuration saved" not in text
    assert "Backup:" not in text  # first run: nothing to back up


def test_non_interactive_init_prints_backup_path(monkeypatch, init_env):
    (init_env / ".env").write_text("ANTHROPIC_API_KEY=old\n")
    result, text = _run_init(monkeypatch, "zh", ("--force",))
    assert result.exit_code == 0, result.output
    assert "备份：" in text
    assert ".env.backup" in text


def test_non_interactive_init_lists_credentials_and_backup_before_next(
    monkeypatch, init_env
):
    (init_env / ".env").write_text("ANTHROPIC_API_KEY=old\n")
    result, text = _run_init(monkeypatch, "en", ("--force",))
    assert result.exit_code == 0, result.output
    credentials = text.index("Credentials:")
    backup = text.index("Backup:")
    next_step = text.index("Next:")
    assert text.index("Workspace:") < credentials < backup < next_step


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_status_not_initialized_survives_markup_in_workspace_path(
    monkeypatch, init_env, lang
):
    cli_module = importlib.import_module("lobster.cli")
    workspace = init_env / "ws[/x][bold]y"
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(workspace))
    monkeypatch.setenv("COLUMNS", "400")
    set_lang(lang)
    result = CliRunner().invoke(cli_module.app, ["status"])
    output = result.output
    assert result.exit_code == 0, output
    assert "Traceback" not in output
    assert "MarkupError" not in output
    assert "ws[/x][bold]y" in output.replace("\n", "")


def _cli_json_stdout(tmp_path, args, lang):
    # A subprocess, not CliRunner: JSON mode rebinds the shared console to the
    # runner's stream, which would leak into later tests in the same process.
    env = os.environ.copy()
    env.update(
        HOME=str(tmp_path),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        PYTHONPATH=str(ROOT),
        LOBSTER_WORKSPACE=str(tmp_path / "ws"),
    )
    env.pop("LOBSTER_LANG", None)
    proc = subprocess.run(  # nosec B603 # fixed argv, no shell
        [sys.executable, "-m", "lobster.cli", "--lang", lang, *args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert payload["success"] is True
    return proc.stdout


@pytest.mark.parametrize(
    "args",
    [["command", "help", "--json"], ["query", "/help", "--json"]],
    ids=["command-help", "query-slash-help"],
)
def test_cli_help_json_stdout_is_identical_across_languages(tmp_path, args):
    english = _cli_json_stdout(tmp_path, args, "en")
    chinese = _cli_json_stdout(tmp_path, args, "zh")
    assert english == chinese
