"""``lobster_docs`` module: search and read the public docs, offline on recorded fixtures."""

import io
import json
import re
import urllib.request
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_docs
from lobster.cli_internal.mcp_docs import DocsError

DATA = Path(__file__).parent / "data" / "docs"
HOST = "https://docs.omics-os.com"
INDEX_URL = f"{HOST}/llm/product-index.json"


class Site:
    """Stands in for the network: serves the recorded index and pages."""

    def __init__(self):
        self.calls = []
        self.online = True

    def __call__(self, url):
        self.calls.append(url)
        if not self.online:
            raise DocsError(f"Could not reach {url}; no network.")
        if url == INDEX_URL:
            return (DATA / "product-index.json").read_bytes()
        m = re.fullmatch(re.escape(HOST) + r"/raw/docs/(.+)\.md", url)
        path = DATA / "raw" / f"{m.group(1)}.md" if m else None
        if path is None or not path.is_file():
            raise DocsError(f"Could not fetch {url}: HTTP 404.")
        return path.read_bytes()


@pytest.fixture
def site(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    fake = Site()
    monkeypatch.setattr(mcp_docs, "_fetch", fake)
    return fake


def results(*args, **kwargs):
    return mcp_docs.search(*args, **kwargs)["results"]


def fences(text):
    return re.findall(r"^```.*?^```", text, flags=re.S | re.M)


# search -------------------------------------------------------------------


def test_search_finds_the_custom_agent_tutorial_and_marks_it_a_use_case(site):
    hits = results("custom agent")
    top = hits[0]
    assert top["slug"] == "tutorials/custom-agent"
    assert top["kind"] == "use_case"
    assert top["url"] == f"{HOST}/docs/tutorials/custom-agent"
    assert top["cached_at"] is None
    assert {"slug", "title", "description", "url", "kind", "matched"} <= set(top)


def test_reference_pages_are_marked_reference(site):
    hits = {h["slug"]: h for h in results("genomics")}
    assert hits["agents/genomics"]["kind"] == "reference"
    assert hits["case-studies/genomics"]["kind"] == "use_case"


def test_search_matches_section_headings(site):
    hits = results("Agent Discovery Pattern")
    assert hits[0]["slug"] == "tutorials/custom-agent"
    assert "Agent Discovery Pattern (v1.0.0+)" in hits[0]["matched"]


def test_search_results_carry_the_reference_notice(site):
    out = mcp_docs.search("genomics")
    assert set(out) == {"results", "notice"}
    assert out["notice"] == mcp_docs.read("agents/genomics")["notice"]
    assert "not instructions" in out["notice"]


def test_control_and_bidi_characters_are_removed_from_listed_text(site, monkeypatch):
    index = json.loads((DATA / "product-index.json").read_text())
    bad = "Ev\u202eil\x1b[31m tit\u200ble\x00"
    index["pages"][1].update(title=bad, description=bad, sections=[bad + " agent"])
    site_fetch = site.__call__
    monkeypatch.setattr(
        mcp_docs,
        "_fetch",
        lambda url: json.dumps(index).encode() if url == INDEX_URL else site_fetch(url),
    )
    slug = index["pages"][1]["slug"]
    hit = next(h for h in results("evil title") if h["slug"] == slug)
    assert hit["title"] == "Evil[31m title"
    for field in (hit["title"], hit["description"], *hit["matched"]):
        assert not re.search(
            r"[\x00-\x08\x0b-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069]",
            field,
        )


def test_search_skips_premium_and_the_intro_page(site):
    slugs = [h["slug"] for h in results("biovault omics-os documentation")]
    assert "cloud/biovault" not in slugs
    assert "" not in slugs


def test_search_limit_is_clamped_and_empty_query_returns_nothing(site):
    assert len(results("agent", limit=1)) == 1
    assert len(results("agent", limit=0)) == 1
    assert len(results("agent", limit=10_000)) <= 20
    assert results("   ") == []
    assert results("zzzzqq") == []


# read ---------------------------------------------------------------------


def test_read_returns_framed_text_for_the_tutorial(site):
    out = mcp_docs.read("tutorials/custom-agent")
    assert out["slug"] == "tutorials/custom-agent"
    assert out["kind"] == "use_case"
    assert out["source_url"] == f"{HOST}/raw/docs/tutorials/custom-agent.md"
    assert out["page"] == 1 and out["pages"] >= 2
    assert "not instructions" in out["notice"]
    assert out["text"].startswith("# Creating Custom Agents Tutorial")
    assert out["cached_at"] is None


def test_pages_join_back_to_the_whole_text_and_keep_fences_whole(site):
    first = mcp_docs.read("tutorials/custom-agent")
    pages = [
        mcp_docs.read("tutorials/custom-agent", page=n)["text"]
        for n in range(1, first["pages"] + 1)
    ]
    whole = "".join(pages)
    source = (DATA / "raw" / "tutorials" / "custom-agent.md").read_text()
    assert fences(whole) == fences(source)
    assert all(len(p) <= mcp_docs.PAGE_CHARS * 2 for p in pages)
    for p in pages:
        assert p.count("```") % 2 == 0, "a code fence was split across pages"


def test_page_out_of_range_is_refused(site):
    for bad in (0, 99):
        with pytest.raises(ValueError, match="page"):
            mcp_docs.read("tutorials/custom-agent", page=bad)


def test_component_blocks_are_stripped_but_prose_stays(site):
    text = mcp_docs.read("agents/genomics")["text"]
    for gone in ("<NextSteps", "<AgentHero", "href:", "import {", "<Rocket"):
        assert gone not in text
    assert "Specialized agent for genomic variant analysis and GWAS." in text
    assert "VCF file loading and parsing" in text


def test_code_fences_are_byte_identical_and_keep_their_tags(site):
    source = (DATA / "raw" / "advanced" / "agent-coordination.md").read_text()
    text = mcp_docs.read("advanced/agent-coordination")["text"]
    pages = mcp_docs.read("advanced/agent-coordination")["pages"]
    text = "".join(
        mcp_docs.read("advanced/agent-coordination", page=n)["text"]
        for n in range(1, pages + 1)
    )
    assert fences(text) == fences(source)
    assert "<Debugging>" in text
    assert "<Mermaid" not in text


def test_strip_unwraps_callout_and_drops_multiline_elements():
    src = (
        'import { X } from "y";\n\n<Callout type="warn">\n  **Note** — kept.\n'
        '</Callout>\n\n<Foo\n  a={[1, 2]}\n  b="x > y"\n/>\n\nafter\n\n'
        "```tsx\n<Foo a={1} />\n```\n"
    )
    out = mcp_docs.strip_mdx(src)
    assert "**Note** — kept." in out
    assert "<Callout" not in out and "</Callout>" not in out
    assert "<Foo\n" not in out and "import" not in out
    assert "after" in out
    assert "```tsx\n<Foo a={1} />\n```" in out


def test_strip_drops_paired_components_with_their_body():
    out = mcp_docs.strip_mdx("a\n\n<Tabs>\n<Tab>\nx\n</Tab>\n</Tabs>\n\nb\n")
    assert "<Tab" not in out and "x" not in out
    assert "a" in out and "b" in out


# the caller never supplies a URL -------------------------------------------


@pytest.mark.parametrize(
    "slug",
    [
        "https://evil.example/x",
        "//evil.example/x",
        "../../etc/passwd",
        "tutorials/../agents/genomics",
        "tutorials/custom-agent.md",
        "Tutorials/Custom-Agent",
        "tutorials/custom-agent?x=1",
        "",
        "not-in-the-index",
        "cloud/biovault",
    ],
)
def test_only_indexed_free_slugs_are_fetched(site, slug):
    with pytest.raises(DocsError):
        mcp_docs.read(slug)
    assert all(c == INDEX_URL for c in site.calls)


# cache ---------------------------------------------------------------------


def test_fresh_cache_answers_without_the_network(site):
    mcp_docs.read("agents/genomics")
    before = len(site.calls)
    mcp_docs.read("agents/genomics")
    results("genomics")
    assert len(site.calls) == before


def test_expired_cache_is_refetched(site, monkeypatch):
    mcp_docs.read("agents/genomics")
    before = len(site.calls)
    real = mcp_docs._now
    monkeypatch.setattr(mcp_docs, "_now", lambda: real() + 24 * 3600 + 5)
    mcp_docs.read("agents/genomics")
    assert len(site.calls) > before


def test_offline_with_an_expired_cache_returns_the_copy_with_its_date(
    site, monkeypatch
):
    live = mcp_docs.read("agents/genomics")
    real = mcp_docs._now
    monkeypatch.setattr(mcp_docs, "_now", lambda: real() + 3 * 24 * 3600)
    site.online = False
    old = mcp_docs.read("agents/genomics")
    assert old["text"] == live["text"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", old["cached_at"])
    assert results("genomics")[0]["cached_at"] == old["cached_at"]


def test_offline_without_a_cache_is_a_one_line_refusal_naming_the_url(site):
    site.online = False
    with pytest.raises(DocsError) as exc:
        results("agent")
    assert INDEX_URL in str(exc.value)
    assert "\n" not in str(exc.value)


def test_cache_lives_under_the_user_cache_dir_and_is_private(site, tmp_path):
    mcp_docs.read("agents/genomics")
    root = tmp_path / "cache" / "omics-os" / "lobster" / "docs"
    assert (root / "index.json").is_file()
    assert any(root.rglob("genomics*"))
    assert (root.stat().st_mode & 0o077) == 0
    assert not list(root.rglob("*.tmp"))


def test_a_corrupt_cache_file_is_ignored(site, tmp_path):
    mcp_docs.read("agents/genomics")
    root = tmp_path / "cache" / "omics-os" / "lobster" / "docs"
    for f in root.rglob("*"):
        if f.is_file():
            f.write_text("{not json")
    assert mcp_docs.read("agents/genomics")["text"]


# transport policy ----------------------------------------------------------


def _redirect(handler, old, new):
    req = urllib.request.Request(old)
    return handler.redirect_request(req, io.BytesIO(), 302, "Found", {}, new)


def test_redirects_stay_on_the_fixed_host_over_https():
    handler = mcp_docs._SameHostRedirect()
    ok = _redirect(handler, f"{HOST}/a", f"{HOST}/b")
    assert ok.full_url == f"{HOST}/b"
    for bad in (
        "https://evil.example/b",
        "http://docs.omics-os.com/b",
        "https://docs.omics-os.com.evil.example/b",
        "https://docs.omics-os.com@evil.example/b",
    ):
        with pytest.raises(DocsError):
            _redirect(handler, f"{HOST}/a", bad)


def test_response_over_the_cap_is_refused():
    class R:
        def __init__(self, n):
            self.buf = io.BytesIO(b"x" * n)

        def read(self, k=-1):
            return self.buf.read(k)

    assert len(mcp_docs._read_capped(R(100), "u")) == 100
    with pytest.raises(DocsError, match="too large"):
        mcp_docs._read_capped(R(mcp_docs.MAX_BYTES + 1), "u")


def test_the_opener_keeps_no_cookies_and_sends_no_credentials():
    opener = mcp_docs._opener()
    names = {type(h).__name__ for h in opener.handlers}
    assert "HTTPCookieProcessor" not in names
    assert "HTTPBasicAuthHandler" not in names
    assert not any(k.lower() == "authorization" for k, _ in opener.addheaders)


def test_fetch_refuses_any_url_off_the_fixed_host():
    for url in (
        "http://docs.omics-os.com/x",
        "https://evil.example/x",
        "file:///etc/passwd",
    ):
        with pytest.raises(DocsError):
            mcp_docs._fetch(url)


def test_module_import_is_light():
    import subprocess
    import sys

    code = (
        "import sys, lobster.cli_internal.mcp_docs as m;"
        "bad=[k for k in sys.modules if k.split('.')[0] in "
        "('langchain','langgraph','requests','httpx')];"
        "print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    ).stdout
    assert out.strip() == "[]"


BAD_INDEXES = {
    "not json": b"{nope",
    "not an object": b"[1]",
    "pages not a list": b'{"pages": "nope"}',
    "entry not an object": b'{"pages": [1]}',
    "slug not a string": b'{"pages": [{"slug": 3, "title": "t"}]}',
    "title not a string": b'{"pages": [{"slug": "a", "title": null}]}',
    "long field": json.dumps(
        {"pages": [{"slug": "a", "title": "t", "description": "x" * 2001}]}
    ).encode(),
    "too many entries": json.dumps(
        {"pages": [{"slug": f"a{i}", "title": "t"} for i in range(2001)]}
    ).encode(),
    "not utf-8": b'{"pages": []}\xff',
}


@pytest.mark.parametrize("name", BAD_INDEXES)
def test_a_bad_index_is_refused_and_never_cached(site, monkeypatch, tmp_path, name):
    monkeypatch.setattr(mcp_docs, "_fetch", lambda url: BAD_INDEXES[name])
    with pytest.raises(DocsError, match="index"):
        results("agent")
    root = tmp_path / "cache"
    assert not [f for f in root.rglob("*") if f.is_file()]


def test_a_bad_index_with_a_cache_falls_back_to_the_stale_copy(site, monkeypatch):
    results("genomics")
    real = mcp_docs._now
    monkeypatch.setattr(mcp_docs, "_now", lambda: real() + 3 * 24 * 3600)
    monkeypatch.setattr(mcp_docs, "_fetch", lambda url: b"{nope")
    hits = results("genomics")
    assert hits and hits[0]["cached_at"]


def test_a_page_that_is_not_utf8_is_refused_and_never_cached(
    site, monkeypatch, tmp_path
):
    results("genomics")
    site_fetch = site.__call__
    monkeypatch.setattr(
        mcp_docs,
        "_fetch",
        lambda url: b"# t\xff\xfe" if url.endswith("genomics.md") else site_fetch(url),
    )
    with pytest.raises(DocsError, match="agents/genomics"):
        mcp_docs.read("agents/genomics")
    assert not list((tmp_path / "cache").rglob("pages/agents/genomics*"))


def test_errors_never_echo_a_response_body(site, monkeypatch):
    secret = "SECRET-BODY-TEXT"
    monkeypatch.setattr(mcp_docs, "_fetch", lambda url: secret.encode())
    with pytest.raises(DocsError) as exc:
        results("agent")
    assert secret not in str(exc.value)
