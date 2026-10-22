
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentMetadata(BaseModel):
    """Metadata shared by all chunks from one document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source: str
    document_type: str = "unknown"
    content_hash: str
    ingested_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    """Validated metadata for an individual text chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    source: str
    document_type: str = "unknown"

    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)

    word_count: int = Field(ge=0)
    char_count: int = Field(ge=0)

    content_hash: str
    ingested_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
