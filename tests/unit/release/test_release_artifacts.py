"""Release regression tests: real archives and mocked official registry responses."""

import base64
import copy
import csv
import hashlib
import importlib.util
import io
import json
import stat
import struct
import tarfile
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "verify_release_artifacts.py"
SPEC = importlib.util.spec_from_file_location("verify_release_artifacts", SCRIPT)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)
VERSION = "1.2.3"
CORE = "lobster-ai"
POLICY = v.platforms()
NAMES = list(v.manifests())


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "tui_builder_test", v.ROOT / "packages/lobster-ai-tui/build_wheel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_rejects_version_override_before_compiling(monkeypatch):
    builder = load_builder()
    monkeypatch.setattr(builder, "get_version", lambda: VERSION)
    compile_binary = Mock()
    monkeypatch.setattr(builder, "build_go_binary", compile_binary)
    monkeypatch.setattr(
        builder.sys,
        "argv",
        ["build_wheel.py", "--platform", "linux-amd64", "--version", "9.9.9"],
    )
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code == 2
    compile_binary.assert_not_called()


def test_builder_reads_version_without_executing_source(tmp_path, monkeypatch):
    builder = load_builder()
    (tmp_path / "lobster").mkdir()
    (tmp_path / "lobster/version.py").write_text(
        f'__version__ = "{VERSION}"\nraise RuntimeError("must not execute")\n'
    )
    monkeypatch.setattr(builder, "REPO_ROOT", tmp_path)
    assert builder.get_version() == VERSION


@pytest.mark.parametrize("content", [b"*", b"*\n"])
def test_uv_build_sentinel_is_not_an_artifact(tmp_path, content):
    make_wheel(tmp_path)
    make_sdist(tmp_path)
    (tmp_path / ".gitignore").write_bytes(content)
    assert set(v.artifacts(VERSION, CORE, tmp_path)) == set(
        v.expected_files(VERSION, CORE)
    )


@pytest.mark.parametrize("content", [b"", b"**", b"secret", b"*\nextra"])
def test_other_hidden_build_contents_are_rejected(tmp_path, content):
    make_wheel(tmp_path)
    make_sdist(tmp_path)
    (tmp_path / ".gitignore").write_bytes(content)
    with pytest.raises(v.VerificationError, match="Unexpected build-directory"):
        v.artifacts(VERSION, CORE, tmp_path)


def test_symlink_build_sentinel_rejected(tmp_path):
    dist = tmp_path / "dist"
    make_wheel(dist)
    make_sdist(dist)
    sentinel_target = tmp_path / "sentinel"
    sentinel_target.write_text("*")
    (dist / ".gitignore").symlink_to(sentinel_target)
    with pytest.raises(v.VerificationError, match="Unexpected build-directory"):
        v.artifacts(VERSION, CORE, dist)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(
        v.urllib.request,
        "build_opener",
        Mock(side_effect=RuntimeError("Unmocked network request")),
    )


@pytest.fixture
def suite(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "lobster").mkdir()
    (root / "lobster/version.py").write_text(f'__version__ = "{VERSION}"\n')
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{CORE}"\n[tool.bumpversion]\ncurrent_version = "{VERSION}"\n'
    )
    for name in NAMES:
        if name != CORE:
            directory = root / "packages" / name
            directory.mkdir(parents=True)
            (directory / "pyproject.toml").write_text(
                f'[project]\nname = "{name}"\nversion = "{VERSION}"\n'
            )
    monkeypatch.setattr(v, "ROOT", root)
    monkeypatch.setattr(v, "platforms", lambda: POLICY)
    return root


def metadata(name, version):
    return f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n".encode()


def binary(platform):
    config = POLICY[platform]
    if config["goos"] == "linux":
        data = bytearray(64)
        data[:7] = b"\x7fELF\x02\x01\x01"
        struct.pack_into(
            "<HHI", data, 16, 2, {"amd64": 62, "arm64": 183}[config["goarch"]], 1
        )
    else:
        data = bytearray(32)
        data[:4] = b"\xcf\xfa\xed\xfe"
        struct.pack_into(
            "<III",
            data,
            4,
            {"amd64": 0x01000007, "arm64": 0x0100000C}[config["goarch"]],
            0,
            2,
        )
    return bytes(data)


def make_wheel(
    directory,
    name=CORE,
    platform=None,
    *,
    version=VERSION,
    metadata_version=None,
    metadata_name=None,
    tags=None,
    binary_data=None,
    binary_mode=0o755,
    omit_binary=False,
    extra=None,
    record_change=None,
):
    expected = v.expected_files(version, name, platform)
    filename = next(n for n in expected if n.endswith(".whl"))
    dist = f"{name.replace('-', '_')}-{version}.dist-info"
    tag = "-".join(filename[:-4].split("-")[-3:])
    if tags is None:
        tags = ["-".join(parts) for parts in sorted(v.expanded_tags(tag))]
    files = {
        f"{dist}/METADATA": metadata(
            metadata_name or name, metadata_version or version
        ),
        f"{dist}/WHEEL": (
            "Wheel-Version: 1.0\n" + "".join(f"Tag: {t}\n" for t in tags)
        ).encode(),
        "module/__init__.py": b"# package\n",
    }
    bin_path = "lobster_ai_tui/bin/lobster-tui"
    if platform and not omit_binary:
        files[bin_path] = binary(platform) if binary_data is None else binary_data
    files.update(extra or {})
    record = f"{dist}/RECORD"
    rows = [
        [
            key,
            "sha256="
            + base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode(),
            str(len(data)),
        ]
        for key, data in files.items()
    ]
    rows.append([record, "", ""])
    if record_change:
        record_change(rows)
    stream = io.StringIO()
    csv.writer(stream).writerows(rows)
    files[record] = stream.getvalue().encode()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, data in files.items():
            info = zipfile.ZipInfo(key)
            info.create_system = 3
            info.external_attr = (
                stat.S_IFREG | (binary_mode if key == bin_path else 0o644)
            ) << 16
            archive.writestr(info, data)
    return path


def make_sdist(
    directory,
    name=CORE,
    *,
    metadata_version=VERSION,
    metadata_name=None,
    extra=None,
    duplicate=False,
    symlink=False,
    omit_metadata=False,
):
    stem = f"{name.replace('-', '_')}-{VERSION}"
    path = directory / f"{stem}.tar.gz"
    directory.mkdir(parents=True, exist_ok=True)
    files = (
        []
        if omit_metadata
        else [(f"{stem}/PKG-INFO", metadata(metadata_name or name, metadata_version))]
    )
    files.extend((extra or {}).items())
    if duplicate:
        files *= 2
    with tarfile.open(path, "w:gz") as archive:
        for key, data in files:
            info = tarfile.TarInfo(key)
            info.size = len(data)
            if symlink:
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                info.size = 0
            archive.addfile(info, io.BytesIO(data))
    return path


def make_artifacts(directory, name=CORE):
    if name == v.TUI:
        for platform in POLICY:
            make_wheel(directory, name, platform)
    else:
        make_wheel(directory, name)
        make_sdist(directory, name)
    return v.artifacts(VERSION, name, directory)


def registry_json(name=CORE, index="testpypi", hashes=None):
    return {
        "info": {"name": name, "version": VERSION, "yanked": False},
        "urls": [
            {
                "filename": filename,
                "url": f"https://{v.INDEXES[index][1]}/packages/ab/cd/{filename}",
                "digests": {
                    "sha256": (
                        hashes[filename]
                        if hashes is not None
                        else hashlib.sha256(filename.encode()).hexdigest()
                    )
                },
                "yanked": False,
                "packagetype": "bdist_wheel" if filename.endswith(".whl") else "sdist",
            }
            for filename in v.expected_files(VERSION, name)
        ],
    }


def mock_registry(monkeypatch, payloads, index="testpypi"):
    def open_url(url, timeout):
        assert timeout == v.TIMEOUT
        prefix = f"https://{v.INDEXES[index][0]}/pypi/"
        assert url.startswith(prefix) and url.endswith(f"/{VERSION}/json")
        name = url[len(prefix) :].split("/")[0]
        response = io.BytesIO(json.dumps(payloads[name]).encode())
        response.geturl = lambda: url
        return response

    opener = Mock()
    opener.open.side_effect = open_url
    monkeypatch.setattr(v.urllib.request, "build_opener", Mock(return_value=opener))
    return opener


def test_source_and_suite(suite):
    assert set(v.source(VERSION)) == set(NAMES)
    assert len(NAMES) == 13
    assert len(POLICY) == 4


@pytest.mark.parametrize(
    "bad", ["1.2", "1.2.3rc1", "v1.2.3", "1.2.3.4", "01.2.3", "1.2.3\n", "1.2.-3"]
)
def test_strict_version(bad):
    with pytest.raises(v.VerificationError):
        v.version_string(bad)


@pytest.mark.parametrize(
    "target",
    ["lobster/version.py", "pyproject.toml", "packages/lobster-ai-tui/pyproject.toml"],
)
def test_stale_source(suite, target):
    path = suite / target
    path.write_text(path.read_text().replace(VERSION, "1.2.2"))
    with pytest.raises(v.VerificationError):
        v.source(VERSION)


def test_source_never_executes(suite):
    (suite / "lobster/version.py").write_text(
        f'raise RuntimeError("must not execute")\n__version__ = "{VERSION}"\n'
    )
    v.source(VERSION)
    (suite / "lobster/version.py").write_text('__version__ = str("1.2.3")\n')
    with pytest.raises(ValueError):
        v.source(VERSION)


def test_missing_manifest(suite):
    (suite / "packages/lobster-ai-tui/pyproject.toml").unlink()
    with pytest.raises(v.VerificationError, match="12 package"):
        v.manifests()


def test_valid_artifacts(suite, tmp_path):
    assert len(make_artifacts(tmp_path / "core")) == 2
    assert len(make_artifacts(tmp_path / "tui", v.TUI)) == 4


@pytest.mark.parametrize("platform", POLICY)
def test_selected_tui(suite, tmp_path, platform):
    make_wheel(tmp_path / "dist", v.TUI, platform)
    assert len(v.artifacts(VERSION, v.TUI, tmp_path / "dist", platform)) == 1
    with pytest.raises(v.VerificationError, match="inventory"):
        v.artifacts(VERSION, v.TUI, tmp_path / "dist")


def test_compressed_tags(suite, tmp_path):
    platform = "linux-amd64"
    tags = [f"py3-none-{POLICY[platform]['wheel_plat']}"]
    make_wheel(tmp_path / "dist", v.TUI, platform, tags=tags)
    v.artifacts(VERSION, v.TUI, tmp_path / "dist", platform)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"metadata_version": "1.2.2"}, "version mismatch"),
        ({"metadata_name": "wrong-name"}, "name mismatch"),
        ({"tags": ["py3-none-linux_x86_64"]}, "tags mismatch"),
        ({"tags": ["not-a-valid-tag"]}, "Malformed wheel tag"),
        ({"tags": ["py3-none-any", "py3-none-any"]}, "Duplicate internal"),
        ({"tags": []}, "tags mismatch"),
        ({"extra": {"../escape": b"oops"}}, "Unsafe archive"),
        ({"extra": {"/absolute": b"oops"}}, "Unsafe archive"),
        ({"extra": {"C:/drive": b"oops"}}, "Unsafe archive"),
        ({"extra": {"module\\escape": b"oops"}}, "Unsafe archive"),
        ({"extra": {"other-1.2.3.dist-info/METADATA": b"oops"}}, "dist-info"),
    ],
)
def test_bad_wheel(suite, tmp_path, options, match):
    path = make_wheel(tmp_path, **options)
    with pytest.raises(v.VerificationError, match=match):
        v.check_wheel(path, CORE, VERSION, None)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda rows: rows[0].__setitem__(1, "sha256=" + "a" * 43), "hash mismatch"),
        (lambda rows: rows[0].__setitem__(2, "99999"), "size mismatch"),
        (lambda rows: rows[0].__setitem__(1, "md5=abc"), "hash algorithm"),
        (lambda rows: rows[0].__setitem__(2, "-1"), "Invalid RECORD size"),
        (lambda rows: rows.pop(0), "membership mismatch"),
        (lambda rows: rows.pop(), "membership mismatch"),
        (lambda rows: rows.append(rows[0]), "Duplicate RECORD"),
        (lambda rows: rows[0].__setitem__(0, "missing"), "missing member"),
        (lambda rows: rows[-1].__setitem__(2, "1"), "self-entry"),
        (lambda rows: rows.append(["bad"]), "Malformed RECORD"),
    ],
)
def test_bad_record(suite, tmp_path, change, match):
    path = make_wheel(tmp_path, record_change=change)
    with pytest.raises(v.VerificationError, match=match):
        v.check_wheel(path, CORE, VERSION, None)


def test_zip_duplicate_and_symlink(suite, tmp_path):
    path = make_wheel(tmp_path)
    with pytest.warns(UserWarning, match="Duplicate"):
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("module/__init__.py", b"other")
    with pytest.raises(v.VerificationError, match="Duplicate wheel"):
        v.check_wheel(path, CORE, VERSION, None)
    path = make_wheel(tmp_path)
    with zipfile.ZipFile(path, "a") as archive:
        info = zipfile.ZipInfo("link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "outside")
    with pytest.raises(v.VerificationError, match="Non-regular"):
        v.check_wheel(path, CORE, VERSION, None)


@pytest.mark.parametrize("platform", POLICY)
@pytest.mark.parametrize(
    "fault", ["architecture", "format", "mode", "missing", "short"]
)
def test_bad_tui(suite, tmp_path, platform, fault):
    options = {}
    if fault == "architecture":
        other = (
            platform.replace("amd64", "arm64")
            if "amd64" in platform
            else platform.replace("arm64", "amd64")
        )
        options["binary_data"] = binary(other)
    elif fault == "format":
        other = "darwin-amd64" if platform.startswith("linux") else "linux-amd64"
        options["binary_data"] = binary(other)
    elif fault == "mode":
        options["binary_mode"] = 0o644
    elif fault == "missing":
        options["omit_binary"] = True
    else:
        options["binary_data"] = b"\x7fELF"
    path = make_wheel(tmp_path, v.TUI, platform, **options)
    with pytest.raises(v.VerificationError):
        v.check_wheel(path, v.TUI, VERSION, platform)


@pytest.mark.parametrize(
    "options",
    [
        {"metadata_version": "1.2.2"},
        {"metadata_name": "other"},
        {"duplicate": True},
        {"symlink": True},
        {"omit_metadata": True},
        {"extra": {"../outside": b"bad"}},
        {"extra": {"unexpected/file": b"bad"}},
    ],
)
def test_bad_sdist(suite, tmp_path, options):
    path = make_sdist(tmp_path, **options)
    with pytest.raises(v.VerificationError):
        v.check_sdist(path, CORE, VERSION)


@pytest.mark.parametrize("fault", ["missing", "extra", "stale", "symlink"])
def test_artifact_inventory(suite, tmp_path, fault):
    directory = tmp_path / "dist"
    make_artifacts(directory)
    wheel = next(directory.glob("*.whl"))
    if fault == "missing":
        wheel.unlink()
    elif fault == "extra":
        (directory / "unrelated.whl").write_bytes(b"extra")
    elif fault == "stale":
        wheel.rename(wheel.with_name(wheel.name.replace(VERSION, "1.2.2")))
    else:
        moved = tmp_path / wheel.name
        wheel.rename(moved)
        wheel.symlink_to(moved)
    with pytest.raises(v.VerificationError):
        v.artifacts(VERSION, CORE, directory)


@pytest.mark.parametrize("index", v.INDEXES)
@pytest.mark.parametrize("name", [CORE, v.TUI])
def test_registry_matches(suite, tmp_path, monkeypatch, index, name):
    directory = tmp_path / "dist"
    hashes = make_artifacts(directory, name)
    opener = mock_registry(
        monkeypatch, {name: registry_json(name, index, hashes)}, index
    )
    v.registry(VERSION, name, directory, index)
    assert opener.open.call_count == 1


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "duplicate",
        "yanked",
        "release-yanked",
        "version",
        "name",
        "digest",
        "type",
    ],
)
def test_registry_inventory(suite, monkeypatch, fault):
    data = registry_json(v.TUI)
    if fault == "missing":
        data["urls"].pop()
    elif fault == "extra":
        extra = copy.deepcopy(data["urls"][0])
        extra["filename"] = "unexpected.whl"
        data["urls"].append(extra)
    elif fault == "duplicate":
        data["urls"].append(data["urls"][0])
    elif fault == "yanked":
        data["urls"][0]["yanked"] = True
    elif fault == "release-yanked":
        data["info"]["yanked"] = True
    elif fault == "version":
        data["info"]["version"] = "1.2.2"
    elif fault == "name":
        data["info"]["name"] = "other"
    elif fault == "digest":
        data["urls"][0]["digests"]["sha256"] = "not-a-hash"
    else:
        data["urls"][0]["packagetype"] = "sdist"
    mock_registry(monkeypatch, {v.TUI: data})
    with pytest.raises(v.VerificationError):
        v.main(["tui-available", VERSION, "--index", "testpypi"])


def test_registry_hash_mismatch(suite, tmp_path, monkeypatch):
    directory = tmp_path / "dist"
    hashes = make_artifacts(directory)
    data = registry_json(hashes=hashes)
    data["urls"][0]["digests"]["sha256"] = "0" * 64
    mock_registry(monkeypatch, {CORE: data})
    with pytest.raises(v.VerificationError, match="SHA256 mismatch"):
        v.registry(VERSION, CORE, directory, "testpypi")


@pytest.mark.parametrize(
    "url",
    [
        "http://test-files.pythonhosted.org/packages/a/{filename}",
        "https://evil.example/packages/a/{filename}",
        "https://test-files.pythonhosted.org.evil.example/packages/a/{filename}",
        "https://user@test-files.pythonhosted.org/packages/a/{filename}",
        "https://test-files.pythonhosted.org:443/packages/a/{filename}",
        "https://files.pythonhosted.org/packages/a/{filename}",
        "https://test-files.pythonhosted.org/packages/a/{filename}#fragment",
        "https://test-files.pythonhosted.org/packages/a/{filename}?query=1",
        "https://test-files.pythonhosted.org/packages/%2e%2e/{filename}",
        "https://test-files.pythonhosted.org/packages/a/wrong.whl",
        "https://test-files.pythonhosted.org/packages/a\n/{filename}",
    ],
)
def test_unsafe_registry_urls(suite, monkeypatch, url):
    data = registry_json()
    entry = data["urls"][0]
    entry["url"] = url.format(filename=entry["filename"])
    mock_registry(monkeypatch, {CORE: data})
    with pytest.raises(v.VerificationError, match="Unsafe"):
        v.registry_files(VERSION, CORE, "testpypi")


def test_http_failure_propagates(suite, monkeypatch):
    opener = Mock()
    opener.open.side_effect = urllib.error.HTTPError(
        "https://test.pypi.org/", 404, "missing", {}, None
    )
    monkeypatch.setattr(v.urllib.request, "build_opener", Mock(return_value=opener))
    with pytest.raises(urllib.error.HTTPError):
        v.main(["tui-available", VERSION, "--index", "testpypi"])
    assert opener.open.call_count == 1


def test_redirect_rejected():
    with pytest.raises(v.VerificationError, match="redirect"):
        v.NoRedirects().redirect_request(
            None, None, 302, "", {}, "https://evil.example"
        )


@pytest.mark.parametrize("index", v.INDEXES)
def test_exact_requirements_all13(suite, tmp_path, monkeypatch, index):
    payloads = {name: registry_json(name, index) for name in NAMES}
    opener = mock_registry(monkeypatch, payloads, index)
    output = tmp_path / "requirements.txt"
    v.main(["requirements", VERSION, "--index", index, "--output", str(output)])
    lines = output.read_text().splitlines()
    assert lines[0] == "--index-url https://pypi.org/simple"
    expected = []
    for name in NAMES:
        wheels = v.expected_files(
            VERSION, name, "linux-amd64" if name == v.TUI else None
        )
        filename = next(n for n in wheels if n.endswith(".whl"))
        entry = next(e for e in payloads[name]["urls"] if e["filename"] == filename)
        label = name + ("[full]" if name == CORE else "")
        expected.append(f"{label} @ {entry['url']}#sha256={entry['digests']['sha256']}")
    assert lines[1:] == expected
    assert len(lines) == 14
    assert opener.open.call_count == 13


def test_requirements_never_partially_written(suite, tmp_path, monkeypatch):
    payloads = {name: registry_json(name) for name in NAMES}
    payloads[NAMES[-1]]["urls"].pop()
    mock_registry(monkeypatch, payloads)
    output = tmp_path / "requirements.txt"
    output.write_text("existing\n")
    with pytest.raises(v.VerificationError):
        v.requirements(VERSION, "testpypi", output)
    assert output.read_text() == "existing\n"


def test_installed(suite, monkeypatch, capsys):
    lookup = Mock(return_value=VERSION)
    monkeypatch.setattr(v.importlib.metadata, "version", lookup)
    assert v.installed(VERSION) == dict.fromkeys(NAMES, VERSION)
    assert set(capsys.readouterr().out.splitlines()) == {
        f"{n}=={VERSION}" for n in NAMES
    }
    assert lookup.call_count == 13
    lookup.side_effect = lambda name: "1.2.2" if name == v.TUI else VERSION
    with pytest.raises(v.VerificationError, match="Installed"):
        v.installed(VERSION)
    assert f"{v.TUI}==1.2.2" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["LOBSTER_AI", "lobster.ai", "lobster__ai"])
def test_normalized_metadata(suite, tmp_path, name):
    wheel = make_wheel(tmp_path, metadata_name=name)
    sdist = make_sdist(tmp_path, metadata_name=name)
    v.check_wheel(wheel, CORE, VERSION, None)
    v.check_sdist(sdist, CORE, VERSION)


def test_incomplete_compressed_tags(suite, tmp_path):
    platform = "linux-amd64"
    tag = POLICY[platform]["wheel_plat"].split(".")[0]
    path = make_wheel(tmp_path, v.TUI, platform, tags=[f"py3-none-{tag}"])
    with pytest.raises(v.VerificationError, match="tags mismatch"):
        v.check_wheel(path, v.TUI, VERSION, platform)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_archive_size_limit(suite, tmp_path, monkeypatch, kind):
    path = make_wheel(tmp_path) if kind == "wheel" else make_sdist(tmp_path)
    monkeypatch.setattr(v, "MAX_ARCHIVE", 1)
    with pytest.raises(v.VerificationError, match="too large"):
        if kind == "wheel":
            v.check_wheel(path, CORE, VERSION, None)
        else:
            v.check_sdist(path, CORE, VERSION)


def test_json_size_limit(suite, monkeypatch):
    mock_registry(monkeypatch, {CORE: registry_json()})
    monkeypatch.setattr(v, "MAX_JSON", 1)
    with pytest.raises(v.VerificationError, match="too large"):
        v.registry_files(VERSION, CORE, "testpypi")


def test_non_tui_platform_rejected(suite):
    with pytest.raises(v.VerificationError, match="only valid"):
        v.expected_files(VERSION, CORE, "linux-amd64")


def test_installed_missing(suite, monkeypatch):
    monkeypatch.setattr(
        v.importlib.metadata,
        "version",
        Mock(side_effect=v.importlib.metadata.PackageNotFoundError("missing")),
    )
    with pytest.raises(v.importlib.metadata.PackageNotFoundError):
        v.installed(VERSION)
