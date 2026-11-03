"""Malformed synthetic helper replies never echo response prose to diagnostics."""

import hashlib
import json
import logging
import traceback
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from lobster.core.exceptions import InferenceRefused
from lobster.services.interaction import component_mapper
from lobster.services.metadata.metadata_validation_service import (
    MetadataValidationService,
    ValidationSeverity,
)
from lobster.utils.json_extractor import extract_json_from_text, safe_json_parse
from lobster.utils.llm_response_utils import extract_text_from_llm_response

MARKER = "SYNTHETIC_RESPONSE_CONTENT_SENTINEL"
REPLY = MARKER + " synthetic non-JSON response"


@pytest.fixture(autouse=True)
def capture_helper_diagnostics(caplog):
    for name in (
        "lobster.utils.json_extractor",
        "lobster.utils.llm_response_utils",
        "lobster.services.metadata.metadata_validation_service",
    ):
        caplog.set_level(logging.DEBUG, logger=name)


@pytest.mark.parametrize("parser", [extract_json_from_text, safe_json_parse])
def test_non_json_exception_and_all_captured_logs_are_content_free(parser, caplog):
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ValueError) as caught:
        parser(REPLY)
    assert MARKER not in str(caught.value)
    assert MARKER not in caplog.text
    assert caplog.records, "Privacy check must observe actual diagnostics"
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


def test_non_json_fallback_is_unchanged_without_prose_logging(caplog):
    caplog.set_level(logging.DEBUG)
    fallback = {"synthetic_fallback": True}
    assert extract_json_from_text(REPLY, fallback) is fallback
    assert MARKER not in caplog.text
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_failed_content_blocks_logs_error_type_only_and_keeps_text_fallback(caplog):
    class SyntheticReply:
        content = REPLY

        @property
        def content_blocks(self):
            raise ValueError(MARKER)

    caplog.set_level(logging.DEBUG)
    assert extract_text_from_llm_response(SyntheticReply()) == REPLY
    assert MARKER not in caplog.text
    assert any("ValueError" in record.getMessage() for record in caplog.records)


def test_diagnostic_hash_does_not_break_fallback_on_lone_unicode_surrogate(caplog):
    caplog.set_level(logging.DEBUG)
    fallback = {"synthetic_fallback": True}
    assert extract_json_from_text(REPLY + chr(0xD800), fallback) is fallback
    assert MARKER not in caplog.text


def test_metadata_non_json_response_keeps_manual_check_without_any_prose_log(caplog):
    caplog.set_level(logging.DEBUG)
    helper = MetadataValidationService()
    model = MagicMock()
    model.invoke.return_value = SimpleNamespace(content=REPLY)
    helper._llm = model
    result = helper.validate_dataset_metadata(
        {
            "title": "Synthetic dataset",
            "samples": {
                "GSM900000000001": {"characteristics_ch1": ["condition: synthetic"]}
            },
        },
        "GSE900000000001",
        ["condition"],
    )
    model.invoke.assert_called_once()
    assert result is not None
    assert result.recommendation == "manual_check"
    assert result.severity == ValidationSeverity.WARNING
    assert result.has_required_fields is False
    assert result.missing_fields == ["condition"]
    assert result.total_samples == 1
    assert result.confidence_score == 0.0
    assert result.warnings == [
        "LLM response parsing failed - manual validation recommended"
    ]
    assert MARKER not in caplog.text
    assert MARKER not in result.model_dump_json()
    assert any(record.levelno == logging.ERROR for record in caplog.records)


@pytest.mark.parametrize(
    "reply",
    [
        '{"synthetic_key": 3}',
        '```json\n{"synthetic_key": 3}\n```',
        'Synthetic introduction\n{"synthetic_key": 3}\nSynthetic footer',
    ],
)
def test_valid_json_extraction_semantics_unchanged(reply):
    assert extract_json_from_text(reply) == {"synthetic_key": 3}


def _structured_reply():
    return {
        "has_required_fields": True,
        "missing_fields": [],
        "available_fields": {"condition": ["synthetic"]},
        "sample_count_by_field": {"condition": 1},
        "total_samples": 1,
        "field_coverage": {"condition": 100.0},
        "recommendation": "proceed",
        "confidence_score": 0.9,
        "warnings": [],
        "blocking_issues": [],
    }


def _validate_reply(reply=None, error=None, n_samples=1):
    helper = MetadataValidationService()
    model = MagicMock()
    helper._llm = model
    if error is not None:
        model.invoke.side_effect = error
    else:
        model.invoke.return_value = SimpleNamespace(content=reply)
    result = helper.validate_dataset_metadata(
        {
            "title": "Synthetic dataset",
            "n_samples": n_samples,
            "samples": {
                "GSM900000000001": {"characteristics_ch1": ["condition: synthetic"]}
            },
        },
        "GSE900000000001",
        ["condition"],
    )
    model.invoke.assert_called_once()
    return helper, result


@pytest.mark.parametrize(
    "kind",
    [
        "confidence_text",
        "confidence_high",
        "confidence_negative",
        "recommendation",
        "available_fields_nested",
        "sample_count_by_field",
        "total_samples",
        "field_coverage",
        "warnings",
        "blocking_issues",
        "missing_fields",
        "severity",
        "surrogate",
        "nonmapping",
    ],
)
def test_schema_failure_diagnostics_and_returned_report_are_content_free(kind, caplog):
    payload = _structured_reply()
    if kind in {"confidence_text", "surrogate"}:
        payload["confidence_score"] = MARKER + (
            chr(0xD800) if kind == "surrogate" else ""
        )
    elif kind == "confidence_high":
        payload["confidence_score"] = 2.0
    elif kind == "confidence_negative":
        payload["confidence_score"] = -0.1
    elif kind == "available_fields_nested":
        payload["available_fields"] = {MARKER: {"nested": MARKER}}
    elif kind == "sample_count_by_field":
        payload["sample_count_by_field"] = {MARKER: MARKER}
    elif kind == "field_coverage":
        payload["field_coverage"] = {MARKER: {"nested": MARKER}}
    elif kind in {"warnings", "blocking_issues", "missing_fields"}:
        payload[kind] = {MARKER: MARKER}
    elif kind == "nonmapping":
        payload = [MARKER]
    else:
        payload[kind] = MARKER
    reply = json.dumps(payload, ensure_ascii=False)
    helper, result = _validate_reply(reply)
    assert result.recommendation == "manual_check"
    assert result.severity == ValidationSeverity.WARNING
    assert result.has_required_fields is False
    assert result.missing_fields == ["condition"]
    assert (
        result.available_fields
        == result.sample_count_by_field
        == result.field_coverage
        == {}
    )
    assert result.total_samples == 1 and result.confidence_score == 0.0
    assert result.blocking_issues == []
    assert result.warnings == [
        "Metadata validation failed - manual validation recommended"
    ]
    assert MARKER not in caplog.text
    assert MARKER not in result.model_dump_json()
    assert MARKER not in helper.format_validation_report(result, "GSE900000000001")
    warning = [
        record
        for record in caplog.records
        if record.name.endswith("metadata_validation_service")
        and record.levelno == logging.WARNING
    ]
    assert len(warning) == 1, "Must observe the actual outer schema/error diagnostic"
    expected_type = "AttributeError" if kind == "nonmapping" else "ValidationError"
    expected_sha = hashlib.sha256(
        reply.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    assert warning[0].getMessage() == (
        f"Metadata validation failed (error_type={expected_type}, "
        f"response_length={len(reply)}, response_sha256={expected_sha})"
    )
    for detail in ("input_value=", "input_type=", "ctx=", "loc="):
        assert detail not in caplog.text and detail not in result.model_dump_json()


def test_error_before_response_extraction_has_absent_text_diagnostic(caplog):
    helper, result = _validate_reply(error=RuntimeError(REPLY))
    assert result.recommendation == "manual_check"
    assert result.warnings == [
        "Metadata validation failed - manual validation recommended"
    ]
    assert MARKER not in caplog.text
    assert MARKER not in helper.format_validation_report(result, "GSE900000000001")
    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        "Metadata validation failed (error_type=RuntimeError, response_length=0, "
        f"response_sha256={hashlib.sha256(b'').hexdigest()})"
    )


@pytest.mark.parametrize("chained", [False, True], ids=["direct", "chained"])
def test_inference_refusal_is_same_object_not_manual_check(chained, caplog):
    refusal = InferenceRefused("policy")
    error = RuntimeError(REPLY) if chained else refusal
    if chained:
        error.__cause__ = refusal
    with pytest.raises(InferenceRefused) as caught:
        _validate_reply(error=error)
    assert caught.value is refusal
    assert MARKER not in caplog.text
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def test_valid_schema_and_ignored_extra_fields_keep_functional_values(caplog):
    payload = _structured_reply()
    payload[MARKER] = {MARKER: MARKER}
    _, result = _validate_reply(json.dumps(payload))
    assert result.has_required_fields is True and result.recommendation == "proceed"
    assert result.severity == ValidationSeverity.CLEAN
    assert result.available_fields == {"condition": ["synthetic"]}
    assert result.sample_count_by_field == {"condition": 1}
    assert result.field_coverage == {"condition": 100.0}
    assert result.total_samples == 1 and result.confidence_score == 0.9
    assert result.warnings == result.blocking_issues == result.missing_fields == []
    assert MARKER not in result.model_dump_json() and MARKER not in caplog.text


def _assert_clean_caller_exception(error, caplog):
    assert error.__context__ is None and error.__cause__ is None
    formatted = "".join(
        traceback.format_exception(type(error), error, error.__traceback__, chain=True)
    )
    assert MARKER not in formatted and MARKER not in caplog.text
    records = [record for record in caplog.records if record.exc_info is not None]
    assert records, "Must exercise actual caller exc_info/exception formatting"
    for record in records:
        assert MARKER not in logging.Formatter().format(record)


def test_invalid_metadata_fallback_error_never_chains_model_response(caplog):
    payload = _structured_reply()
    payload["confidence_score"] = MARKER
    error = None
    try:
        _validate_reply(json.dumps(payload), n_samples=None)
    except ValidationError as caught:
        error = caught
        logging.getLogger("synthetic_privacy_caller").exception(
            "Synthetic caller failure"
        )
    assert error is not None  # Invalid caller-side metadata still raises.
    _assert_clean_caller_exception(error, caplog)


def _component_error():
    with pytest.raises(ValidationError) as caught:
        component_mapper.AskUserContext.model_validate(
            {"fields": [{"key": "answer", "type": MARKER}]}
        )
    return caught.value


def _component_model(error):
    model = MagicMock()
    model.with_structured_output.return_value.invoke.side_effect = error
    return model


def test_component_structured_schema_failure_logs_type_only_and_keeps_fallback(caplog):
    caplog.set_level(logging.DEBUG, logger=component_mapper.__name__)
    model = _component_model(_component_error())
    result = component_mapper.map_question("Interpret this output", llm=model)
    assert [field.model_dump(exclude_defaults=True) for field in result.fields] == [
        {"key": "answer", "type": "text", "question": "Interpret this output"}
    ]
    records = [
        record for record in caplog.records if record.name == component_mapper.__name__
    ]
    assert len(records) == 1 and records[0].levelno == logging.DEBUG
    assert records[0].exc_info is None
    assert "ValidationError" in records[0].getMessage()
    assert "response_length=0" in records[0].getMessage()
    assert hashlib.sha256(b"").hexdigest() in records[0].getMessage()
    assert MARKER not in caplog.text and MARKER not in result.model_dump_json()


def test_component_fallback_failure_never_chains_model_response(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=component_mapper.__name__)
    model = _component_model(_component_error())

    def failed_fallback(**_):
        raise ValueError("Synthetic caller-only fallback failure")

    monkeypatch.setattr(component_mapper, "AskUserContext", failed_fallback)
    error = None
    try:
        component_mapper.map_question("Interpret this output", llm=model)
    except ValueError as caught:
        error = caught
        logging.getLogger("synthetic_privacy_caller").exception(
            "Synthetic caller failure"
        )
    assert error is not None
    _assert_clean_caller_exception(error, caplog)


@pytest.mark.parametrize("chained", [False, True])
def test_component_inference_refusal_keeps_same_object(chained, caplog):
    caplog.set_level(logging.DEBUG, logger=component_mapper.__name__)
    refusal = InferenceRefused("policy")
    error = RuntimeError(REPLY) if chained else refusal
    if chained:
        error.__cause__ = refusal
    with pytest.raises(InferenceRefused) as caught:
        component_mapper.map_question(
            "Interpret this output", llm=_component_model(error)
        )
    assert caught.value is refusal
    assert MARKER not in caplog.text
    assert not [
        record for record in caplog.records if record.name == component_mapper.__name__
    ]


def test_missing_expected_keys_semantics_unchanged():
    with pytest.raises(ValueError, match="Missing expected keys"):
        safe_json_parse('{"synthetic_key": 3}', ["synthetic_missing"])
