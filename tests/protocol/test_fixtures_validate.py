"""Every recorded protocol fixture validates against the engine schema."""

import pytest

from scripts.validate_ndjson import load_validator, validate_lines
from tests.protocol.stdio_driver import SCHEMAS, STDIO


@pytest.mark.parametrize(
    "fixture", sorted(STDIO.glob("*.ndjson")), ids=lambda p: p.name
)
def test_fixture_is_schema_valid(fixture):
    validator = load_validator(SCHEMAS / "protocol-v1.schema.json")
    with fixture.open(encoding="utf-8") as handle:
        assert validate_lines(validator, fixture.name, handle) == []
