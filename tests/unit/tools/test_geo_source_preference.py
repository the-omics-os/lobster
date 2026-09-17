from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

from lobster.core.download_queue import DownloadQueue
from lobster.core.schemas.download_queue import DownloadQueueEntry
from lobster.tools.geo_source_preference import ask_geo_source_preference


@pytest.mark.parametrize(
    "answer,expected",
    [("Author-uploaded GEO data", "author"), ("NCBI-generated counts", "ncbi")],
)
def test_preference_interrupt_and_resume_before_save(tmp_path, answer, expected):
    queue = DownloadQueue(tmp_path / "queue.jsonl")

    class State(TypedDict):
        source: str

    def prepare(state):
        source = ask_geo_source_preference("GSE164073")
        queue.add_entry(
            DownloadQueueEntry(
                entry_id="test",
                dataset_id="GSE164073",
                database="geo",
                has_ncbi_rnaseq_counts=True,
                selected_source=source,
                source_preference_answered=True,
                matrix_url="https://example.org/author.tsv",
            )
        )
        return {"source": source}

    builder = StateGraph(State)
    builder.add_node("prepare", prepare)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test"}}
    result = graph.invoke({"source": ""}, config)
    assert result["__interrupt__"][0].value["component"] == "select"
    assert queue.list_entries() == []
    result = graph.invoke(Command(resume={"selected": answer}), config)
    assert result["source"] == expected
    entry = queue.get_entry("test")
    assert entry.selected_source == expected
    assert entry.matrix_url == "https://example.org/author.tsv"
    assert len(queue.list_entries()) == 1


def test_queue_preference_update_preserves_execution(tmp_path):
    queue = DownloadQueue(tmp_path / "queue.jsonl")
    entry = DownloadQueueEntry(
        entry_id="test", dataset_id="GSE1", database="geo", matrix_url="author"
    )
    queue.add_entry(entry)
    queue.update_geo_preference("test", True, "ncbi", True)
    updated = queue.get_entry("test")
    assert updated.selected_source == "ncbi"
    assert updated.matrix_url == entry.matrix_url
    assert updated.status == entry.status


def test_non_geo_defaults_and_update_rejection(tmp_path):
    queue = DownloadQueue(tmp_path / "queue.jsonl")
    entry = DownloadQueueEntry(entry_id="test", dataset_id="PXD1", database="pride")
    queue.add_entry(entry)
    assert entry.selected_source is None
    with pytest.raises(ValueError, match="only to GEO"):
        queue.update_geo_preference("test", True, "ncbi", True)
    assert queue.get_entry("test").selected_source is None


def test_handoff_appends_persisted_geo_preference(tmp_path):
    import asyncio
    from unittest.mock import AsyncMock
    from langchain_core.messages import AIMessage
    from lobster.agents.graph import _invoke_and_store

    queue = DownloadQueue(tmp_path / "queue.jsonl")
    queue.add_entry(
        DownloadQueueEntry(
            entry_id="geo",
            dataset_id="GSE164073",
            database="geo",
            has_ncbi_rnaseq_counts=True,
            selected_source="ncbi",
            source_preference_answered=True,
        )
    )
    agent = Mock()
    agent.ainvoke = AsyncMock(
        return_value={
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "prepare_dataset_download",
                            "args": {"accession": "GSE164073"},
                            "id": "call1",
                        }
                    ],
                ),
                AIMessage(content="Prepared your dataset."),
            ]
        }
    )
    dm = Mock(download_queue=queue)
    result = asyncio.run(
        _invoke_and_store(agent, "research_agent", "prepare", None, dm)
    )
    assert "has_ncbi_rnaseq_counts=True" in result
    assert "selected_source=ncbi" in result
    other = asyncio.run(
        _invoke_and_store(agent, "data_expert_agent", "prepare", None, dm)
    )
    assert other == "Prepared your dataset."


def test_source_choice_happens_after_metadata_before_entry():
    from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer
    from lobster.core.schemas.download_queue import StrategyConfig

    preparer = GEOQueuePreparer(Mock())
    events = []
    preparer.fetch_metadata = Mock(
        side_effect=lambda accession: (events.append("metadata") or {}, None)
    )
    preparer._geo_provider = Mock()
    preparer._geo_provider.has_ncbi_rnaseq_counts.side_effect = lambda accession: (
        events.append("availability") or True
    )
    preparer.source_selector = lambda accession: events.append("question") or "ncbi"
    preparer.extract_download_urls = Mock(
        return_value=Mock(to_queue_entry_fields=lambda: {}, file_count=0)
    )
    preparer.recommend_strategy = Mock(
        return_value=StrategyConfig(
            strategy_name="AUTO",
            concatenation_strategy="auto",
            confidence=0.5,
            rationale="test",
        )
    )
    result = preparer.prepare_queue_entry("GSE164073")
    assert events == ["metadata", "availability", "question"]
    assert result.queue_entry.selected_source == "ncbi"
    assert result.queue_entry.has_ncbi_rnaseq_counts is True
