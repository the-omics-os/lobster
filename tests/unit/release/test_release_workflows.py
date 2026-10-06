"""Offline release gates; use RELEASE_TEST_BASH for an Actions-compatible Bash."""

import json
import os
import shutil

# Execute repository workflow fixtures with isolated tools and no host credentials.
import subprocess  # nosec B404
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
PUBLISHERS = ("publish-packages", "publish-tui")
BASH = os.environ.get("RELEASE_TEST_BASH") or shutil.which("bash")


def workflow(name):
    # BaseLoader preserves the Actions `on` key instead of treating it as True.
    # BaseLoader constructs only strings/containers; no Python object constructors.
    return yaml.load(  # nosec B506
        (WORKFLOWS / f"{name}.yml").read_text(), Loader=yaml.BaseLoader
    )


SUITE = workflow("publish-packages")
TUI = workflow("publish-tui")
RELEASE = workflow("public-release")
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["bumpversion"][
    "current_version"
]
DISTRIBUTIONS = {
    tomllib.loads(path.read_text())["project"]["name"]
    for path in [ROOT / "pyproject.toml", *ROOT.glob("packages/*/pyproject.toml")]
}


def step(job, **match):
    matches = [s for s in job["steps"] if all(s.get(k) == v for k, v in match.items())]
    assert len(matches) == 1, match
    return matches[0]


def validation(name):
    key = "validate-version" if name == "publish-packages" else "validate"
    return workflow(name)["jobs"][key]


# These executables never delegate to a host git/gh or read host credentials.
STUB = """import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
config = json.loads(os.environ['STUB_CONFIG'])
with open(os.environ['CALL_LOG'], 'a') as log:
    log.write(json.dumps([name, *args]) + '\\n')
if name == 'git':
    tag = 'v' + os.environ['EXPECTED_VERSION'] + '-test'
    if args == ['fetch', 'origin', f'refs/tags/{tag}:refs/tags/{tag}']:
        sys.exit(config.get('fetch_status', 0))
    assert args == ['rev-parse', tag + '^{commit}'], args
    print(config.get('test_sha', os.environ['GITHUB_SHA']))
elif name == 'gh':
    target = args[args.index('--workflow') + 1]
    assert target in ('publish-packages.yml', 'publish-tui.yml'), args
    waiter = config.get('waiter', False)
    branch = os.environ['GITHUB_REF_NAME'] if waiter else 'v' + os.environ['EXPECTED_VERSION'] + '-test'
    query = 'length' if waiter else '.[0].databaseId // empty'
    assert args == ['run', 'list', '--repo', os.environ['GITHUB_REPOSITORY'],
                    '--workflow', target, '--branch', branch,
                    '--commit', os.environ['GITHUB_SHA'], '--status', 'success',
                    '--json', 'databaseId', '--jq', query], args
    if config.get('gh_status'):
        sys.exit(config['gh_status'])
    run_id = {'publish-packages.yml': '123', 'publish-tui.yml': '456'}[target]
    print(config.get(target, '1' if waiter else run_id))
elif name == 'python':
    assert config.get('waiter'), 'Only the waiter uses a fake verifier'
    assert args == ['scripts/verify_release_artifacts.py', 'tui-available',
                    os.environ['VERSION'], '--index', os.environ['INDEX']], args
    sys.exit(config.get('inventory_status', 0))
elif name == 'seq':
    assert args == ['1', '30'], args
    print('\\n'.join(map(str, range(1, 31))))
elif name == 'sleep':
    assert args == ['30'], args
else:
    raise AssertionError(name)
"""


class Shell:
    def __init__(self, directory):
        self.directory = directory
        self.bin = directory / "bin"
        self.bin.mkdir()
        for name in ("git", "gh", "seq", "sleep"):
            self.stub(name)
        (self.bin / "python").symlink_to(sys.executable)
        self.log = directory / "calls.jsonl"
        self.output = directory / "outputs"
        self.summary = directory / "summary.md"
        self.env = {
            "PATH": str(self.bin),
            "HOME": str(directory),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REPOSITORY": "the-omics-os/lobster",
            "GITHUB_SHA": "a" * 40,
            "GITHUB_REF_NAME": f"v{VERSION}-test",
            "GITHUB_REF": f"refs/tags/v{VERSION}-test",
            "GITHUB_OUTPUT": str(self.output),
            "GITHUB_STEP_SUMMARY": str(self.summary),
            "REQUEST_VERSION": VERSION,
            "REQUEST_TARGET": "testpypi",
            "EXPECTED_VERSION": VERSION,
            "CALL_LOG": str(self.log),
            "VERSION": VERSION,
        }

    def stub(self, name):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n{STUB}")
        path.chmod(0o755)

    def run(self, script, *, config=None, **env):
        # Repository-owned shell fixtures; PATH contains only our tools/stubs,
        # HOME is temporary, and no host credentials are passed to subprocesses.
        return subprocess.run(  # nosec B603
            [BASH, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
            cwd=ROOT,
            env={**self.env, **env, "STUB_CONFIG": json.dumps(config or {})},
            capture_output=True,
            text=True,
            timeout=15,
        )

    @property
    def calls(self):
        return (
            [json.loads(line) for line in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )

    @property
    def outputs(self):
        return (
            dict(line.split("=", 1) for line in self.output.read_text().splitlines())
            if self.output.exists()
            else {}
        )


@pytest.fixture
def shell(tmp_path):
    return Shell(tmp_path)


@pytest.fixture(params=PUBLISHERS)
def publisher(request):
    return request.param


def validate(shell, publisher, *, production=False, config=None, **env):
    tag = f"v{VERSION}" + ("" if production else "-test")
    defaults = {
        "GITHUB_REF_NAME": tag,
        "GITHUB_REF": f"refs/tags/{tag}",
        "REQUEST_TARGET": "pypi" if production else "testpypi",
    }
    return shell.run(
        step(validation(publisher), id="version")["run"],
        config=config,
        **{**defaults, **env},
    )


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
@pytest.mark.parametrize("production", [False, True])
def test_valid_tag_uses_real_source_verifier(shell, publisher, event, production):
    result = validate(shell, publisher, production=production, GITHUB_EVENT_NAME=event)
    assert result.returncode == 0, result.stderr
    assert shell.outputs["version"] == VERSION
    assert shell.outputs["is_test"] == str(not production).lower()
    if production:
        assert [call[0] for call in shell.calls] == ["git", "git", "gh", "gh"]
        assert {call[call.index("--workflow") + 1] for call in shell.calls[2:]} == {
            f"{name}.yml" for name in PUBLISHERS
        }
        assert shell.outputs["test_run_id"] == (
            "123" if publisher == "publish-packages" else "456"
        )
        if publisher == "publish-packages":
            assert shell.outputs["tui_test_run_id"] == "456"
    else:
        assert shell.calls == []
        assert set(shell.outputs) == {"version", "is_test"}


@pytest.mark.parametrize(
    "config", [{"fetch_status": 1}, {"test_sha": ""}, {"test_sha": "b" * 40}]
)
def test_production_requires_matching_test_commit(shell, publisher, config):
    result = validate(shell, publisher, production=True, config=config)
    assert result.returncode != 0
    assert shell.calls and all(call[0] == "git" for call in shell.calls)


@pytest.mark.parametrize("source", PUBLISHERS)
@pytest.mark.parametrize(
    "run_id", ["", "0", "01", "null", "not-a-run", "123\n456", "$(touch invalid)"]
)
def test_production_requires_both_successful_run_ids(shell, publisher, source, run_id):
    result = validate(
        shell, publisher, production=True, config={f"{source}.yml": run_id}
    )
    assert result.returncode != 0
    assert any(source + ".yml" in call for call in shell.calls)


def test_promotion_api_failure_is_fatal(shell, publisher):
    assert (
        validate(shell, publisher, production=True, config={"gh_status": 1}).returncode
        != 0
    )


@pytest.mark.parametrize(
    "ref,target",
    [
        ("refs/heads/main", "testpypi"),
        ("refs/heads/main", "pypi"),
        ("refs/tags/v{version}-test", "pypi"),
        ("refs/tags/v{version}", "testpypi"),
        ("refs/tags/v0.0.0-test", "testpypi"),
        ("refs/tags/v{version}-test", "other"),
    ],
)
def test_dispatch_requires_matching_tag_and_target(shell, publisher, ref, target):
    ref = ref.format(version=VERSION)
    result = validate(
        shell,
        publisher,
        GITHUB_EVENT_NAME="workflow_dispatch",
        GITHUB_REF=ref,
        GITHUB_REF_NAME=ref.rsplit("/", 1)[-1],
        REQUEST_TARGET=target,
    )
    assert result.returncode != 0
    assert shell.calls == []


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
@pytest.mark.parametrize(
    "version",
    [
        "",
        "v1.2.3",
        "01.2.3",
        "1.2",
        "1.2.3rc1",
        "1.2.3\n",
        "999999.0.0",
        "$(: > {marker})",
        "1.2.3; : > {marker}",
    ],
)
def test_invalid_versions_are_data_not_shell(shell, publisher, event, version):
    marker = shell.directory / "injected"
    version = version.format(marker=marker)
    result = validate(
        shell,
        publisher,
        GITHUB_EVENT_NAME=event,
        REQUEST_VERSION=version,
        GITHUB_REF_NAME=f"v{version}-test",
        GITHUB_REF=f"refs/tags/v{version}-test",
    )
    assert result.returncode != 0
    assert not marker.exists()
    assert shell.calls == []


def test_complete_suite_is_required_before_release():
    jobs = SUITE["jobs"]
    names = DISTRIBUTIONS - {"lobster-ai-tui"}
    publishers = {f"publish-{name}" for name in names}
    assert len(names) == 12
    assert {
        row["name"] for row in jobs["build"]["strategy"]["matrix"]["package"]
    } == names
    assert {key for key in jobs if key.startswith("publish-")} == publishers
    verify = jobs["verify-installation"]
    assert set(verify["needs"]) == publishers | {"validate-version"}
    assert publishers <= set(jobs["summary"]["needs"])
    for key in publishers | {"verify-installation"}:
        assert (
            "if" not in jobs[key]
        ), f"{key} must require successful dependencies on both registries"
        assert jobs[key].get("continue-on-error", "false") == "false"
    for key in publishers:
        required = (
            "wait-for-tui" if key == "publish-lobster-ai" else "publish-lobster-ai"
        )
        assert set(jobs[key]["needs"]) == {"validate-version", "build", required}
    scripts = "\n".join(s.get("run", "") for s in verify["steps"])
    for command in (" registry ", " requirements ", " installed ", "uv pip check"):
        assert command in scripts


def test_build_and_promotion_are_mutually_exclusive(publisher):
    document = workflow(publisher)
    key = "validate-version" if publisher == "publish-packages" else "validate"
    build = document["jobs"]["build"]
    assert build["needs"] == key
    build_steps = [
        s
        for s in build["steps"]
        if "uv build" in s.get("run", "") or "build_wheel.py" in s.get("run", "")
    ]
    promotions = [s for s in build["steps"] if "run-id" in s.get("with", {})]
    assert len(build_steps) == 1 and promotions
    assert build_steps[0]["if"] == f"needs.{key}.outputs.is_test == 'true'"
    for promotion in promotions:
        assert promotion["uses"].startswith("actions/download-artifact@")
        assert promotion["if"] == f"needs.{key}.outputs.is_test == 'false'"
        assert (
            promotion["with"]["run-id"]
            == "${{ needs." + key + ".outputs.test_run_id }}"
        )
    assert any(
        "verify_release_artifacts.py artifacts" in s.get("run", "") and "if" not in s
        for s in build["steps"]
    )
    comparison_job = (
        build if publisher == "publish-packages" else document["jobs"]["publish"]
    )
    comparisons = [
        s for s in comparison_job["steps"] if "--index testpypi" in s.get("run", "")
    ]
    assert len(comparisons) == 1
    assert comparisons[0]["if"] == f"needs.{key}.outputs.is_test == 'false'"
    assert " registry " in comparisons[0]["run"]


def assert_release_tests_are_required(document):
    quality = document["jobs"]["quality-and-tests"]
    contract = step(quality, id="release-contracts")
    assert contract.get("continue-on-error", "false") == "false"
    assert quality.get("continue-on-error", "false") == "false"
    assert contract["if"] == "${{ !cancelled() && steps.install.outcome == 'success' }}"
    command = contract["run"].replace("\\\n", " ")
    assert "python -m pytest tests/unit/release " in command
    assert "--confcutdir=tests/unit/release" in command
    assert "--junitxml=release-contract-results.xml" in command
    assert (
        "scripts/check_test_results.py --require-no-skips release-contract-results.xml"
        in command
    )
    assert "quality-and-tests" in document["jobs"]["ci-summary"]["needs"]
    upload = step(quality, name="Upload release contract results")
    assert upload["with"]["path"] == "release-contract-results.xml"
    assert "${{ github.run_attempt }}" in upload["with"]["name"]


def test_basic_ci_requires_release_directory_and_nonempty_unskipped_evidence():
    assert_release_tests_are_required(workflow("ci-basic"))


@pytest.mark.parametrize(
    "mutation", ["omit-directory", "allow-skips", "ignore-failure"]
)
def test_release_ci_gate_rejects_weakened_selection(mutation):
    document = workflow("ci-basic")
    contract = step(document["jobs"]["quality-and-tests"], id="release-contracts")
    if mutation == "omit-directory":
        contract["run"] = contract["run"].replace(
            "tests/unit/release", "tests/unit/config"
        )
    elif mutation == "allow-skips":
        contract["run"] = contract["run"].replace("--require-no-skips", "")
    else:
        contract["continue-on-error"] = "true"
    with pytest.raises(AssertionError):
        assert_release_tests_are_required(document)


@pytest.mark.parametrize("job_name", ["fast-validation", "extended-tests"])
def test_uv_validation_does_not_enable_an_unused_pip_cache(job_name):
    job = workflow("pr-validation-basic")["jobs"][job_name]
    python_setup = step(job, name="Set up Python 3.12")
    assert "cache" not in python_setup["with"]
    assert step(job, name="Set up uv")["uses"].startswith("astral-sh/setup-uv@")


def test_verification_artifact_names_are_retry_safe():
    upload = step(
        SUITE["jobs"]["verify-installation"],
        name="Retain exact registry installation pins",
    )
    assert "${{ github.run_attempt }}" in upload["with"]["name"]
    assert upload["with"]["if-no-files-found"] == "error"


def test_single_release_owner_and_scoped_permissions():
    assert set(RELEASE["on"]) == {"workflow_call"}
    callers = []
    for path in WORKFLOWS.glob("*.yml"):
        # BaseLoader cannot instantiate arbitrary objects, unlike Loader.
        document = yaml.load(path.read_text(), Loader=yaml.BaseLoader)  # nosec B506
        for key, job in (document.get("jobs") or {}).items():
            if job.get("uses") == "./.github/workflows/public-release.yml":
                callers.append((path.stem, key))
    assert callers == [("publish-packages", "github-release")]
    caller = SUITE["jobs"]["github-release"]
    assert set(caller["needs"]) == {"validate-version", "verify-installation"}
    # Without always()/failure(), Actions also requires successful dependencies.
    assert caller["if"] == "needs.validate-version.outputs.is_test == 'false'"
    assert (
        caller["with"]["tui-test-run-id"]
        == "${{ needs.validate-version.outputs.tui_test_run_id }}"
    )
    assert (
        RELEASE["permissions"]
        == caller["permissions"]
        == {"contents": "write", "actions": "read"}
    )
    for document in (SUITE, TUI):
        assert document["permissions"] == {"contents": "read", "actions": "read"}
        assert set(document["on"]) == {"push", "workflow_dispatch"}
        for key, job in document["jobs"].items():
            publishing = any(
                s.get("uses", "").startswith("pypa/gh-action-pypi-publish@")
                for s in job.get("steps", [])
            )
            if publishing:
                assert job["permissions"] == {"contents": "read", "id-token": "write"}
                assert (
                    "testpypi-" in job["environment"]["name"]
                    and "'pypi-" in job["environment"]["name"]
                )
            elif key != "github-release":
                assert not any(
                    value == "write" for value in job.get("permissions", {}).values()
                )
    assert not any("release" in key for key in TUI["jobs"])
    for job in TUI["jobs"].values():
        for action in job.get("steps", []):
            assert "gh release" not in action.get("run", "")
            assert "release" not in action.get("uses", "").lower()
    assert {
        row["platform"] for row in TUI["jobs"]["build"]["strategy"]["matrix"]["include"]
    } == {"linux-amd64", "linux-arm64", "darwin-amd64", "darwin-arm64"}


@pytest.mark.parametrize(
    "inventory_status,run_count,success",
    [(0, "1", True), (1, "1", False), (0, "0", False), (0, "invalid", False)],
)
def test_waiter_requires_inventory_and_successful_tui_run(
    shell, inventory_status, run_count, success
):
    # Only this polling test replaces the registry verifier; validation above is real.
    (shell.bin / "python").unlink()
    shell.stub("python")
    job = SUITE["jobs"]["wait-for-tui"]
    assert 0 < int(job["timeout-minutes"]) <= 20
    result = shell.run(
        step(job, name="Require all four TUI wheels")["run"],
        config={
            "waiter": True,
            "inventory_status": inventory_status,
            "publish-tui.yml": run_count,
        },
        INDEX="testpypi",
    )
    assert (result.returncode == 0) is success, result.stderr
    attempts = [call for call in shell.calls if call[0] == "python"]
    assert len(attempts) == (1 if success else 30)
    if inventory_status:
        assert not any(call[0] == "gh" for call in shell.calls)


@pytest.mark.parametrize("is_test", [False, True])
@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_summary_reports_actual_outcomes(shell, is_test, result):
    job = SUITE["jobs"]["summary"]
    assert job["if"] == "always()"
    outcomes = {name: {"result": "success"} for name in job["needs"]}
    if is_test:
        outcomes["github-release"]["result"] = "skipped"
    outcomes["verify-installation"]["result"] = result
    completed = shell.run(
        step(job, name="Generate truthful release summary")["run"],
        NEEDS_JSON=json.dumps(outcomes),
        IS_TEST=str(is_test).lower(),
    )
    assert (completed.returncode == 0) is (result == "success"), completed.stderr
    summary = shell.summary.read_text()
    assert f"## {'TestPyPI' if is_test else 'PyPI'} release {VERSION}" in summary
    for name, outcome in outcomes.items():
        assert f"| {name} | {outcome['result']} |" in summary


@pytest.mark.parametrize("job_name", SUITE["jobs"]["summary"]["needs"])
def test_summary_rejects_any_skipped_production_dependency(shell, job_name):
    job = SUITE["jobs"]["summary"]
    outcomes = {name: {"result": "success"} for name in job["needs"]}
    outcomes[job_name]["result"] = "skipped"
    result = shell.run(
        step(job, name="Generate truthful release summary")["run"],
        NEEDS_JSON=json.dumps(outcomes),
        IS_TEST="false",
    )
    assert result.returncode != 0
    assert job_name in result.stderr
    assert f"| {job_name} | skipped |" in shell.summary.read_text()
