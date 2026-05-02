"""Artifact metadata contract for vector collection compatibility."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ArtifactMetadata(BaseModel):
    """Describes a pre-built vector collection artifact.

    Used to verify runtime embedder compatibility before querying.
    If the runtime embedder doesn't match the artifact's embedding config,
    queries against that collection should fail closed.
    """

    embedding_provider: str = Field(
        description="Provider that built embeddings (sapbert, minilm, openai)"
    )
    model_id: str = Field(description="Specific model ID used for embedding")
    dimensions: int = Field(description="Embedding vector dimensionality")
    collection: str = Field(
        description="Collection name (e.g. mondo_v2024_01)"
    )
    collection_version: str = Field(
        description="Version tag of the source ontology"
    )
    build_hash: str = Field(description="SHA256 of source OWL/OBO file")
    build_date: str = Field(description="ISO 8601 build timestamp")


class CollectionUnavailable:
    """Returned when a collection cannot be queried safely."""

    def __init__(
        self,
        collection: str,
        reason: str,
        expected: ArtifactMetadata | None = None,
        actual_provider: str | None = None,
    ):
        self.collection = collection
        self.reason = reason
        self.expected = expected
        self.actual_provider = actual_provider

    def __repr__(self) -> str:
        return f"CollectionUnavailable(collection={self.collection!r}, reason={self.reason!r})"
