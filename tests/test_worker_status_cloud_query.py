from lobster.cli_internal.commands.light.cloud_query import (
    CloudStreamResult,
    _parse_datastream_line,
)


def test_parse_worker_status_state_patch_and_clear() -> None:
    result = CloudStreamResult()
    seen: list[dict | None] = []

    _parse_datastream_line(
        'aui-state:[{"type":"set","path":["worker_status"],"value":{"_v":1,"session_id":"s1","run_id":"r1","transport":"ecs","status":"starting","phase":"launch","message":"Launching remote worker"}}]',
        result,
        on_worker_status=seen.append,
    )

    assert result.worker_status == {
        "_v": 1,
        "session_id": "s1",
        "run_id": "r1",
        "transport": "ecs",
        "status": "starting",
        "phase": "launch",
        "message": "Launching remote worker",
    }
    assert seen == [result.worker_status]

    _parse_datastream_line(
        'aui-state:[{"type":"set","path":["worker_status"],"value":null}]',
        result,
        on_worker_status=seen.append,
    )

    assert result.worker_status is None
    assert seen[-1] is None
