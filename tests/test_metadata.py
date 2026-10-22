
import pytest
from pydantic import ValidationError

from eddie_twin.metadata import ChunkMetadata
from eddie_twin.ingestion.document_identity import (
    calculate_file_hash,
    create_document_id,
)


def test_document_id_is_stable_for_same_hash():
    file_hash = "a" * 64

    assert create_document_id(file_hash) == create_document_id(file_hash)


def test_document_id_rejects_invalid_hash():
    with pytest.raises(ValueError):
        create_document_id("invalid")


def test_file_hash_is_content_based(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "renamed.pdf"

    first.write_bytes(b"same document content")
    second.write_bytes(b"same document content")

    assert calculate_file_hash(first) == calculate_file_hash(second)


def test_chunk_metadata_validates_page_number():
    with pytest.raises(ValidationError):
        ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc_example",
            source="cv.pdf",
            page_number=0,
            chunk_index=0,
            word_count=10,
            char_count=50,
            content_hash="abc123",
        )
