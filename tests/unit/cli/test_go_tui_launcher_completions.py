from pathlib import Path

from lobster.cli_internal import go_tui_launcher


class _FakeBridge:
    def __init__(self):
        self.calls = []

    def send(self, msg_type, payload=None, msg_id=""):
        self.calls.append((msg_type, payload or {}, msg_id))


class _FakeClient:
    def __init__(self, available_datasets=None):
        self.data_manager = type(
            "DM",
            (),
            {"available_datasets": available_datasets or {}},
        )()


def test_completion_request_read_prefixes_command(tmp_path, monkeypatch):
    (tmp_path / "data.csv").write_text("x")
    (tmp_path / "datasets").mkdir()
    monkeypatch.chdir(tmp_path)

    bridge = _FakeBridge()
    client = _FakeClient()
    event = {
        "id": "req-1",
        "payload": {"command": "/read", "prefix": "da"},
    }

    go_tui_launcher._handle_completion_request(bridge, client, event)

    assert bridge.calls
    msg_type, payload, msg_id = bridge.calls[-1]
    assert msg_type == "completion_response"
    assert msg_id == "req-1"
    assert "/read data.csv" in payload["suggestions"]
    assert "/read datasets/" in payload["suggestions"]


def test_completion_request_quotes_paths_with_spaces(tmp_path, monkeypatch):
    (tmp_path / "my data.csv").write_text("x")
    monkeypatch.chdir(tmp_path)

    bridge = _FakeBridge()
    client = _FakeClient()
    event = {
        "id": "req-2",
        "payload": {"command": "/read", "prefix": "my"},
    }

    go_tui_launcher._handle_completion_request(bridge, client, event)

    _, payload, _ = bridge.calls[-1]
    assert '/read "my data.csv"' in payload["suggestions"]


def test_completion_request_workspace_load_includes_dataset_names():
    bridge = _FakeBridge()
    client = _FakeClient(available_datasets={"rna_reference": {}, "proteomics_v1": {}})
    event = {
        "id": "req-3",
        "payload": {"command": "/workspace load", "prefix": "rn"},
    }

    go_tui_launcher._handle_completion_request(bridge, client, event)

    _, payload, _ = bridge.calls[-1]
    assert "/workspace load rna_reference" in payload["suggestions"]
    assert all(s.startswith("/workspace load ") for s in payload["suggestions"])
