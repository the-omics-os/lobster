#!/usr/bin/env python3
"""Verify release sources, archives and exact official-registry inventories.

No archive extraction, binary execution, retries or package installation occurs here.
The requirements output pins every first-party wheel; resolve third-party dependencies
with production PyPI alone (never add TestPyPI as an extra index).
"""

import argparse
import ast
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import itertools
import json
import re
import stat
import struct
import tarfile
import tomllib
import urllib.parse
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUI = "lobster-ai-tui"
TIMEOUT = 30
MAX_JSON = 4 * 1024 * 1024
MAX_ARCHIVE = 512 * 1024 * 1024
MAX_MEMBERS = 100_000
INDEXES = {
    "pypi": ("pypi.org", "files.pythonhosted.org"),
    "testpypi": ("test.pypi.org", "test-files.pythonhosted.org"),
}


class VerificationError(ValueError):
    """An input does not satisfy the release contract."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def version_string(value):
    require(
        isinstance(value, str)
        and re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", value),
        f"Expected strict X.Y.Z version, got {value!r}",
    )
    return value


def normalized_name(name):
    require(
        isinstance(name, str)
        and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name),
        f"Invalid distribution name: {name!r}",
    )
    return re.sub(r"[-_.]+", "-", name).lower()


def platforms():
    path = ROOT / "packages" / TUI / "build_wheel.py"
    spec = importlib.util.spec_from_file_location("release_tui_builder", path)
    require(spec is not None and spec.loader is not None, "Cannot load TUI builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PLATFORMS


def manifests():
    paths = sorted((ROOT / "packages").glob("*/pyproject.toml"))
    require(len(paths) == 12, f"Expected 12 package manifests, found {len(paths)}")
    result = {}
    for path in [ROOT / "pyproject.toml", *paths]:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        name = normalized_name(data["project"]["name"])
        require(name not in result, f"Duplicate distribution: {name}")
        result[name] = data
    return result


def source(version):
    version_string(version)
    suite = manifests()
    tree = ast.parse((ROOT / "lobster" / "version.py").read_text(encoding="utf-8"))
    values = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == "__version__" for t in targets):
            values.append(ast.literal_eval(node.value))
    require(
        values == [version], f"lobster/version.py: expected {version}, got {values}"
    )
    root_name = next(iter(suite))
    for name, data in suite.items():
        actual = (
            data["tool"]["bumpversion"]["current_version"]
            if name == root_name
            else data["project"]["version"]
        )
        require(actual == version, f"{name}: expected {version}, got {actual}")
    return list(suite)


def expected_files(version, name, platform=None):
    version_string(version)
    name = normalized_name(name)
    require(name in manifests(), f"Unknown suite distribution: {name}")
    stem = f"{name.replace('-', '_')}-{version}"
    if name != TUI:
        require(platform is None, "--platform is only valid for lobster-ai-tui")
        return {f"{stem}-py3-none-any.whl": None, f"{stem}.tar.gz": None}
    policy = platforms()
    require(platform is None or platform in policy, f"Unknown platform: {platform}")
    return {
        f"{stem}-py3-none-{'.'.join(sorted(config['wheel_plat'].split('.')))}.whl": key
        for key, config in policy.items()
        if platform is None or key == platform
    }


def safe_member(name):
    require(isinstance(name, str) and name, "Empty archive member")
    require(
        not any(c in name for c in "\\\x00\r\n:")
        and not name.startswith("/")
        and all(part not in ("", ".", "..") for part in name.rstrip("/").split("/")),
        f"Unsafe archive member: {name!r}",
    )
    return name.rstrip("/")


def check_metadata(raw, name, version):
    message = BytesParser().parsebytes(raw)
    require(len(message.get_all("Name", [])) == 1, "Expected one metadata Name")
    require(len(message.get_all("Version", [])) == 1, "Expected one metadata Version")
    require(
        normalized_name(message["Name"]) == normalized_name(name),
        "Metadata name mismatch",
    )
    require(message["Version"] == version, "Metadata version mismatch")


def expanded_tags(tag):
    require(
        re.fullmatch(
            r"[a-z0-9_]+(?:\.[a-z0-9_]+)*-[a-z0-9_]+(?:\.[a-z0-9_]+)*-[a-z0-9_]+(?:\.[a-z0-9_]+)*",
            tag,
        ),
        f"Malformed wheel tag: {tag!r}",
    )
    return set(itertools.product(*(part.split(".") for part in tag.split("-"))))


def check_binary(data, mode, platform):
    require(mode & 0o111, "TUI binary is not executable")
    config = platforms()[platform]
    if config["goos"] == "linux":
        require(
            len(data) >= 64 and data[:7] == b"\x7fELF\x02\x01\x01",
            "Expected little-endian ELF64 TUI",
        )
        kind, machine = struct.unpack_from("<HH", data, 16)
        expected = {"amd64": 62, "arm64": 183}[config["goarch"]]
        require(kind in (2, 3) and machine == expected, "ELF target mismatch")
    else:
        require(config["goos"] == "darwin", "Unsupported binary format")
        require(
            len(data) >= 32 and data[:4] == b"\xcf\xfa\xed\xfe",
            "Expected little-endian Mach-O64 TUI",
        )
        cpu = struct.unpack_from("<I", data, 4)[0]
        expected = {"amd64": 0x01000007, "arm64": 0x0100000C}[config["goarch"]]
        require(
            cpu == expected and struct.unpack_from("<I", data, 12)[0] == 2,
            "Mach-O target mismatch",
        )


def check_wheel(path, name, version, platform):
    dist_info = f"{normalized_name(name).replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(len(infos) <= MAX_MEMBERS, "Too many wheel members")
        require(sum(i.file_size for i in infos) <= MAX_ARCHIVE, "Wheel is too large")
        seen = set()
        files = {}
        for info in infos:
            member = safe_member(info.orig_filename)
            require(member not in seen, f"Duplicate wheel member: {member}")
            seen.add(member)
            mode = info.external_attr >> 16
            require(
                stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR),
                "Non-regular wheel member",
            )
            if not info.is_dir():
                require(not stat.S_ISDIR(mode), "Directory mode on wheel file")
                files[info.filename] = info
        require(
            {p.split("/")[0] for p in files if p.split("/")[0].endswith(".dist-info")}
            == {dist_info},
            "Unexpected wheel dist-info directory",
        )
        metadata, wheel, record = (
            f"{dist_info}/{leaf}" for leaf in ("METADATA", "WHEEL", "RECORD")
        )
        require({metadata, wheel, record} <= files.keys(), "Missing wheel metadata")
        check_metadata(archive.read(metadata), name, version)
        headers = BytesParser().parsebytes(archive.read(wheel))
        tags = headers.get_all("Tag", [])
        internal = set()
        for tag in tags:
            expanded = expanded_tags(tag)
            require(
                not internal.intersection(expanded), "Duplicate internal wheel tags"
            )
            internal.update(expanded)
        filename_tags = expanded_tags("-".join(path.name[:-4].split("-")[-3:]))
        require(internal == filename_tags, "Wheel filename/internal tags mismatch")
        rows = list(
            csv.reader(io.StringIO(archive.read(record).decode("utf-8")), strict=True)
        )
        recorded = set()
        for row in rows:
            require(len(row) == 3, "Malformed RECORD row")
            member, digest, size = row
            safe_member(member)
            require(member not in recorded, f"Duplicate RECORD member: {member}")
            recorded.add(member)
            require(member in files, f"RECORD references missing member: {member}")
            if member == record:
                require(
                    digest == size == "", "RECORD self-entry must have empty hash/size"
                )
                continue
            require(re.fullmatch(r"0|[1-9][0-9]*", size), "Invalid RECORD size")
            require(
                int(size) == files[member].file_size, f"RECORD size mismatch: {member}"
            )
            algorithm, separator, value = digest.partition("=")
            require(
                separator and algorithm in ("sha256", "sha384", "sha512"),
                "Invalid RECORD hash algorithm",
            )
            actual = (
                base64.urlsafe_b64encode(
                    hashlib.new(algorithm, archive.read(member)).digest()
                )
                .rstrip(b"=")
                .decode("ascii")
            )
            require(value == actual, f"RECORD hash mismatch: {member}")
        require(recorded == files.keys(), "RECORD membership mismatch")
        if platform is not None:
            binary = "lobster_ai_tui/bin/lobster-tui"
            require(binary in files, "Missing TUI binary")
            check_binary(
                archive.read(binary), files[binary].external_attr >> 16, platform
            )


def check_sdist(path, name, version):
    prefix = path.name[:-7]
    seen = set()
    metadata = None
    total = 0
    # Streaming mode bounds header processing as well as advertised file sizes.
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            key = safe_member(member.name)
            require(key not in seen, f"Duplicate sdist member: {key}")
            seen.add(key)
            total += member.size
            require(
                len(seen) <= MAX_MEMBERS and total <= MAX_ARCHIVE, "Sdist is too large"
            )
            require(
                key == prefix or key.startswith(prefix + "/"), "Unexpected sdist root"
            )
            require(member.isfile() or member.isdir(), "Non-regular sdist member")
            if key == f"{prefix}/PKG-INFO":
                require(member.isfile(), "PKG-INFO is not a file")
                stream = archive.extractfile(member)
                require(stream is not None, "Missing PKG-INFO data")
                metadata = stream.read()
    require(metadata is not None, "Missing sdist PKG-INFO")
    check_metadata(metadata, name, version)


def artifacts(version, name, directory, platform=None):
    expected = expected_files(version, name, platform)
    directory = Path(directory)
    entries = list(directory.iterdir())
    # uv build creates this exact sentinel; it is not a distribution artifact.
    sentinel = directory / ".gitignore"
    if sentinel in entries:
        require(
            sentinel.is_file()
            and not sentinel.is_symlink()
            and sentinel.stat().st_size <= 2
            and sentinel.read_bytes() in (b"*", b"*\n"),
            "Unexpected build-directory .gitignore",
        )
        entries.remove(sentinel)
    require(
        {p.name for p in entries} == expected.keys(),
        f"Artifact inventory mismatch: expected {sorted(expected)}, got {sorted(p.name for p in entries)}",
    )
    hashes = {}
    for path in entries:
        require(
            path.is_file() and not path.is_symlink(),
            f"Not a regular artifact: {path.name}",
        )
        require(path.stat().st_size <= MAX_ARCHIVE, "Artifact is too large")
        if path.suffix == ".whl":
            check_wheel(path, name, version, expected[path.name])
        else:
            check_sdist(path, name, version)
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise VerificationError("Registry redirects are not allowed")


def safe_file_url(url, filename, index):
    require(isinstance(url, str) and not re.search(r"[\s\\]", url), "Unsafe file URL")
    parsed = urllib.parse.urlsplit(url)
    require(
        parsed.scheme == "https"
        and parsed.netloc == INDEXES[index][1]
        and not parsed.query
        and not parsed.fragment
        and parsed.path.startswith("/packages/")
        and urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1]) == filename,
        f"Unsafe file URL: {url!r}",
    )
    safe_member(urllib.parse.unquote(parsed.path).lstrip("/"))
    return url


def registry_files(version, name, index):
    expected = expected_files(version, name)
    name = normalized_name(name)
    url = f"https://{INDEXES[index][0]}/pypi/{name}/{version}/json"
    opener = urllib.request.build_opener(NoRedirects())
    with opener.open(url, timeout=TIMEOUT) as response:
        require(response.geturl() == url, "Unexpected registry response URL")
        raw = response.read(MAX_JSON + 1)
    require(len(raw) <= MAX_JSON, "Registry JSON is too large")
    data = json.loads(raw)
    require(normalized_name(data["info"]["name"]) == name, "Registry name mismatch")
    require(data["info"]["version"] == version, "Registry version mismatch")
    require(data["info"].get("yanked", False) is False, "Registry release is yanked")
    result = {}
    for entry in data["urls"]:
        filename = entry["filename"]
        require(
            filename in expected and filename not in result,
            f"Unexpected or duplicate registry file: {filename}",
        )
        require(entry["yanked"] is False, f"Yanked registry file: {filename}")
        require(
            entry["packagetype"]
            == ("bdist_wheel" if filename.endswith(".whl") else "sdist"),
            "Registry package type mismatch",
        )
        digest = entry["digests"]["sha256"]
        require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            "Invalid registry SHA256",
        )
        safe_file_url(entry["url"], filename, index)
        result[filename] = entry
    require(result.keys() == expected.keys(), "Registry inventory mismatch")
    return result


def registry(version, name, directory, index):
    local = artifacts(version, name, directory)
    remote = registry_files(version, name, index)
    require(
        local == {name: entry["digests"]["sha256"] for name, entry in remote.items()},
        "Registry/local SHA256 mismatch",
    )


def requirements(version, index, output):
    version_string(version)
    lines = ["--index-url https://pypi.org/simple"]
    for name in manifests():
        entries = registry_files(version, name, index)
        wheels = expected_files(version, name, "linux-amd64" if name == TUI else None)
        filename = next(filename for filename in wheels if filename.endswith(".whl"))
        entry = entries[filename]
        reference = name + ("[full]" if name == "lobster-ai" else "")
        lines.append(
            f"{reference} @ {entry['url']}#sha256={entry['digests']['sha256']}"
        )
    # Write only after every distribution has passed, never emit partial pins.
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def installed(version):
    version_string(version)
    actual = {name: importlib.metadata.version(name) for name in manifests()}
    for name, found in actual.items():
        print(f"{name}=={found}")
    require(
        all(found == version for found in actual.values()),
        "Installed suite version mismatch",
    )
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "source",
        "artifacts",
        "registry",
        "tui-available",
        "requirements",
        "installed",
    ):
        sub = commands.add_parser(command)
        sub.add_argument("version", type=version_string)
        if command in ("artifacts", "registry"):
            sub.add_argument("name")
            sub.add_argument("directory", type=Path)
        if command == "artifacts":
            sub.add_argument("--platform", choices=list(platforms()))
        if command in ("registry", "tui-available", "requirements"):
            sub.add_argument("--index", required=True, choices=list(INDEXES))
        if command == "requirements":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "source":
        for name in source(args.version):
            print(f"{name}=={args.version}")
    elif args.command == "artifacts":
        artifacts(args.version, args.name, args.directory, args.platform)
    elif args.command == "registry":
        registry(args.version, args.name, args.directory, args.index)
    elif args.command == "tui-available":
        registry_files(args.version, TUI, args.index)
    elif args.command == "requirements":
        requirements(args.version, args.index, args.output)
    else:
        installed(args.version)


if __name__ == "__main__":
    main()
