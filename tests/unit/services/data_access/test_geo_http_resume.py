"""A GEO file over a slow line that drops the connection is finished, not restarted.

A local HTTP server plays NCBI: it cuts the connection part-way through the body, may
ignore Range, or may change the file between requests.
"""

import gzip
import random
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest

from lobster.services.data_access.geo import downloader as dl
from lobster.services.data_access.geo.downloader import GEODownloadManager

BODY = gzip.compress(random.Random(1).randbytes(300_000))  # a real gzip file, ~300 KB


class _Server:
    """Serves ``body``; ``script`` holds one entry per GET, then the last one repeats."""

    def __init__(self, body, script, etag='"v1"'):
        self.body, self.script, self.etag = body, list(script), etag
        self.requests = []  # (Range header or None, If-Range header or None)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                n = len(outer.requests)
                step = outer.script[min(n, len(outer.script) - 1)]
                rng, if_range = self.headers.get("Range"), self.headers.get("If-Range")
                outer.requests.append((rng, if_range))
                body, etag = step.get("body", outer.body), step.get("etag", outer.etag)
                start = 0
                if rng and step.get("range", True) and if_range in (None, etag):
                    start = int(rng.split("=")[1].rstrip("-"))
                send = body[start:]
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(send)))
                if etag:
                    self.send_header("ETag", etag)
                if start:
                    self.send_header(
                        "Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}"
                    )
                self.end_headers()
                cut = step.get("cut")
                self.wfile.write(send if cut is None else send[:cut])
                self.wfile.flush()
                if step.get("stall"):
                    time.sleep(step["stall"])
                if cut is not None or step.get("stall"):
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/f.gz"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def _start(range_header):
    return int(range_header.split("=")[1].rstrip("-"))


@pytest.fixture
def serve():
    servers = []

    def make(script, body=BODY, **kw):
        servers.append(_Server(body, script, **kw))
        return servers[-1]

    yield make
    for s in servers:
        s.httpd.shutdown()
        s.httpd.server_close()


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("LOBSTER_DOWNLOAD_IDLE_TIMEOUT_S", "1")
    m = GEODownloadManager(cache_dir=str(tmp_path / "cache"))
    m.session.trust_env = False  # no proxy for 127.0.0.1
    return m


def test_a_dropped_connection_resumes_with_a_range_request(serve, manager, tmp_path):
    server = serve([{"cut": 40_000}, {"cut": 50_000}, {}])
    target = tmp_path / "f.gz"
    assert manager.download_file(server.url, target) is True
    assert target.read_bytes() == BODY
    first, second = (_start(r[0]) for r in server.requests[1:])
    assert 0 < first <= 40_000 and first < second <= 90_000
    assert server.requests[0][0] is None
    assert not list(tmp_path.glob("*.part"))


def test_a_server_that_ignores_range_restarts_the_file(serve, manager, tmp_path):
    server = serve([{"cut": 40_000}, {"range": False}])
    target = tmp_path / "f.gz"
    assert manager.download_file(server.url, target) is True
    assert target.read_bytes() == BODY  # not 1000 bytes + a whole second copy


def test_a_file_that_changed_between_requests_is_not_glued_together(
    serve, manager, tmp_path
):
    newer = gzip.compress(b"a different file" * 5000)
    server = serve([{"cut": 40_000}, {"etag": '"v2"', "body": newer}])
    target = tmp_path / "f.gz"
    # If-Range does not match: the server sends the whole current file, kept whole
    assert manager.download_file(server.url, target) is True
    assert target.read_bytes() == newer
    assert 0 < _start(server.requests[1][0]) <= 40_000
    assert server.requests[1][1] == '"v1"'


def test_it_gives_up_after_attempts_that_gain_nothing(serve, manager, tmp_path):
    server = serve([{"cut": 0}])
    target = tmp_path / "f.gz"
    assert manager.download_file(server.url, target) is False
    assert not target.exists()
    assert len(server.requests) <= 6  # bounded, not a retry storm


def test_a_stall_longer_than_the_idle_bound_counts_as_a_dropped_connection(
    serve, manager, tmp_path
):
    server = serve([{"cut": 40_000, "stall": 3}, {}])  # idle bound is 1 s here
    target = tmp_path / "f.gz"
    assert manager.download_file(server.url, target) is True
    assert target.read_bytes() == BODY
    assert 0 < _start(server.requests[1][0]) <= 40_000


def test_progress_is_logged_per_file(serve, manager, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_PROGRESS_LOG_INTERVAL_S", 0)
    log = Mock()
    monkeypatch.setattr(dl, "logger", log)
    server = serve([{}])
    assert manager.download_file(server.url, tmp_path / "f.gz") is True
    lines = [str(c.args[0]) % c.args[1:] for c in log.info.call_args_list]
    assert any("f.gz" in line and "KB/s" in line for line in lines), lines
