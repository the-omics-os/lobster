"""A GEO file over a slow line that drops the connection is finished, not restarted.

A local HTTP server plays NCBI: it cuts the connection part-way through the body, may
ignore Range, or may change the file between requests.
"""

import ftplib
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
from lobster.utils import download_activity

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
                data = send if cut is None else send[:cut]
                pace = step.get("pace")  # a trickle: 1 KB, a pause, 1 KB ...
                for i in range(0, len(data), 1024 if pace else max(len(data), 1)):
                    self.wfile.write(data[i : i + (1024 if pace else len(data))])
                    self.wfile.flush()
                    if pace:
                        time.sleep(pace)
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


def test_a_resumed_file_that_fails_the_gzip_check_is_fetched_once_more_whole(
    serve, manager, tmp_path
):
    bad_tail = BODY[:30_000] + b"\xff" * (len(BODY) - 30_000)  # same size, damaged
    server = serve([{"cut": 40_000}, {"body": bad_tail}, {}])
    target = tmp_path / "f.gz"
    assert manager.download_file(server.url, target) is True
    assert target.read_bytes() == BODY
    assert len(server.requests) == 3 and server.requests[2][0] is None


def test_the_download_in_progress_is_visible_to_a_host_that_polls(
    serve, manager, tmp_path
):
    server = serve([{"cut": 40_000, "stall": 2}, {}])
    target = tmp_path / "f.gz"
    done = threading.Thread(target=manager.download_file, args=(server.url, target))
    done.start()
    seen = None
    deadline = time.monotonic() + 5
    while seen is None and time.monotonic() < deadline:
        seen = download_activity.snapshot()
        time.sleep(0.05)
    done.join(10)
    assert seen is not None and seen["file"] == "f.gz"
    assert seen["mb_total"] == round(len(BODY) / 1e6, 1) and seen["mb_done"] >= 0
    assert "kb_per_s" in seen
    assert download_activity.snapshot() is None  # nothing once the file is done


def test_a_cancel_ends_a_download_in_progress_without_retries(serve, manager, tmp_path):
    server = serve([{"cut": 40_000, "stall": 3}, {}])
    target, result, stop = tmp_path / "f.gz", [], threading.Event()
    download_activity.set_stop(stop)
    try:
        worker = threading.Thread(
            target=lambda: result.append(manager.download_file(server.url, target))
        )
        worker.start()
        time.sleep(0.3)
        stop.set()
        worker.join(5)  # the idle bound is 1 s here
        assert not worker.is_alive() and result == [False]
        assert len(server.requests) == 1  # no resume, no fallback
        assert manager.download_file(server.url, target) is False
        assert len(server.requests) == 1  # nothing is started after the cancel
    finally:
        download_activity.set_stop(None)


class _FakeFTP:
    """An FTP server that serves BODY; ``script`` has one entry per RETR, the last repeats."""

    script: list = []
    retrs: list = []  # the REST offset of each RETR

    def connect(self, host, port=21, timeout=None):
        pass

    def login(self, *args):
        pass

    def size(self, path):
        return len(BODY)

    def voidcmd(self, cmd):
        pass

    def quit(self):
        pass

    def retrbinary(self, cmd, callback, blocksize=8192, rest=None):
        step = self.script[min(len(self.retrs), len(self.script) - 1)]
        self.retrs.append(rest)
        if rest and step.get("no_rest"):
            raise ftplib.error_perm("502 REST not implemented")
        data, cut = BODY[rest or 0 :], step.get("cut")
        end = len(data) if cut is None else cut
        for i in range(0, end, blocksize):
            callback(data[i : min(i + blocksize, end)])
        if step.get("cancel"):
            step["cancel"].set()  # the host cancels while the transfer runs
        if cut is not None:
            raise ConnectionResetError("dropped")


@pytest.fixture
def ftp(monkeypatch):
    _FakeFTP.retrs = []
    monkeypatch.setattr(dl.ftplib, "FTP", _FakeFTP)
    monkeypatch.setattr(dl.time, "sleep", lambda _s: None)
    return _FakeFTP


FTP_URL = "ftp://files.example.org/geo/f.gz"


def test_a_dropped_ftp_transfer_resumes_with_rest(ftp, manager, tmp_path):
    ftp.script = [{"cut": 40_000}, {"cut": 30_000}, {}]
    target = tmp_path / "f.gz"
    assert manager.download_file(FTP_URL, target) is True
    assert target.read_bytes() == BODY
    assert ftp.retrs[0] is None and 0 < ftp.retrs[1] < ftp.retrs[2]
    assert not list(tmp_path.glob("*.part"))


def test_a_server_without_rest_gets_the_file_again_from_zero(ftp, manager, tmp_path):
    ftp.script = [{"cut": 40_000}, {"no_rest": True}, {}]
    target = tmp_path / "f.gz"
    assert manager.download_file(FTP_URL, target) is True
    assert target.read_bytes() == BODY  # not 40 KB + a whole second copy
    assert ftp.retrs[-1] is None


def test_a_cancel_ends_an_ftp_transfer_without_retries(ftp, manager, tmp_path):
    stop = threading.Event()
    ftp.script = [{"cut": 40_000, "cancel": stop}, {}]
    download_activity.set_stop(stop)
    try:
        assert manager.download_file(FTP_URL, tmp_path / "f.gz") is False
        assert len(ftp.retrs) == 1  # no second attempt
    finally:
        download_activity.set_stop(None)


def test_the_slow_dropping_stalling_changing_ncbi_pattern_ends_with_the_whole_file(
    serve, manager, tmp_path
):
    # What a slow NCBI line did to a real run: a trickle that drops mid-body again and
    # again, one stall, and the file replaced on the server part-way through.
    newer = gzip.compress(random.Random(2).randbytes(300_000))
    v2 = {"pace": 0.005, "etag": '"v2"', "body": newer}
    server = serve(
        [
            {"pace": 0.005, "cut": 30_000},
            {"pace": 0.005, "cut": 20_000, "stall": 3},
            {**v2, "cut": 15_000},
            {**v2, "cut": 25_000},
            v2,
        ]
    )
    target, seen = tmp_path / "f.gz", []
    worker = threading.Thread(target=manager.download_file, args=(server.url, target))
    worker.start()
    while worker.is_alive():
        seen.append(download_activity.snapshot())
        time.sleep(0.05)
    assert target.read_bytes() == newer  # the file as it is now, never a splice
    assert not list(tmp_path.glob("*.part"))
    assert any(s and s["file"] == "f.gz" for s in seen)  # a host could see it run
    assert len(server.requests) >= 5
