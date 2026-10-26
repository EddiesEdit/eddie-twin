
import pytest
import math
from eddie_twin.vectorstores.qdrant_store import QdrantVectorStore


@pytest.fixture
def store(tmp_path):
    vector_store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_documents",
        vector_size=3,
    )
    yield vector_store
    vector_store.close()


def make_chunk(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc_test",
        "source": "cv.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "text": text,
    }


def test_upsert_and_search_returns_relevant_chunk(store):
    chunks = [
        make_chunk("chunk_1", "I am a machine learning engineer."),
        make_chunk("chunk_2", "I enjoy cooking and gardening."),
    ]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    count = store.upsert_chunks(chunks, vectors)
    results = store.search([0.99, 0.01, 0.0], top_k=1)

    assert count == 2
    assert len(results) == 1
    assert results[0]["text"] == "I am a machine learning engineer."
    assert results[0]["metadata"]["source"] == "cv.pdf"
    assert results[0]["score"] > 0.9


def test_empty_upsert_returns_zero(store):
    assert store.upsert_chunks([], []) == 0


def test_upsert_rejects_mismatched_chunk_and_vector_counts(store):
    chunks = [make_chunk("chunk_1", "Example text.")]

    with pytest.raises(ValueError, match="exactly one vector"):
        store.upsert_chunks(chunks, [])


def test_upsert_rejects_wrong_vector_dimensions(store):
    chunks = [make_chunk("chunk_1", "Example text.")]

    with pytest.raises(ValueError, match="Expected vector size"):
        store.upsert_chunks(chunks, [[0.1, 0.2]])


def test_search_rejects_wrong_vector_dimensions(store):
    with pytest.raises(ValueError, match="Expected query vector size"):
        store.search([0.1, 0.2])


def test_search_rejects_invalid_top_k(store):
    with pytest.raises(ValueError, match="top_k must be positive"):
        store.search([1.0, 0.0, 0.0], top_k=0)


def test_chunk_ids_produce_deterministic_point_ids():
    first = QdrantVectorStore._point_id("doc_test:page-1:chunk-0")
    second = QdrantVectorStore._point_id("doc_test:page-1:chunk-0")

    assert first == second

@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_upsert_rejects_non_finite_vector_values(store, invalid_value):
    chunks = [make_chunk("chunk_invalid", "Example text.")]
    vectors = [[1.0, invalid_value, 0.0]]

    with pytest.raises(ValueError, match="finite"):
        store.upsert_chunks(chunks, vectors)


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_search_rejects_non_finite_query_values(store, invalid_value):
    query_vector = [1.0, invalid_value, 0.0]

    with pytest.raises(ValueError, match="finite"):
        store.search(query_vector)