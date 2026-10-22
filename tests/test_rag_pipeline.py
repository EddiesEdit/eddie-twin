
from pathlib import Path

from eddie_twin.rag.pipeline import RAGPipeline


class FakeIngestionService:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.calls = []

    def ingest_pdf(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "document": {
                "document_id": "doc_test",
                "source": Path(kwargs["file_path"]).name,
            },
            "chunks": self.chunks,
        }


class FakeEmbeddings:
    def __init__(self, vectors=None):
        self.vectors = vectors or []
        self.received_texts = []

    def embed_texts(self, texts):
        self.received_texts.extend(texts)
        return self.vectors


class FakeVectorStore:
    def __init__(self, stored_count=None):
        self.stored_count = stored_count
        self.received_chunks = []
        self.received_vectors = []
        self.closed = False

    def upsert_chunks(self, chunks, vectors):
        self.received_chunks = chunks
        self.received_vectors = vectors
        return (
            len(chunks)
            if self.stored_count is None
            else self.stored_count
        )

    def close(self):
        self.closed = True


def make_chunk(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc_test",
        "source": "cv.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "text": text,
    }


def make_pipeline(chunks, vectors, stored_count=None):
    ingestion = FakeIngestionService(chunks)
    embeddings = FakeEmbeddings(vectors)
    store = FakeVectorStore(stored_count)

    pipeline = RAGPipeline(
        ingestion_service=ingestion,
        embeddings=embeddings,
        vector_store=store,
    )

    return pipeline, ingestion, embeddings, store


def test_pipeline_embeds_and_stores_chunks():
    chunks = [
        make_chunk("chunk_1", "I build machine learning systems."),
        make_chunk("chunk_2", "I work with Python."),
    ]
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    pipeline, _, embeddings, store = make_pipeline(chunks, vectors)

    result = pipeline.ingest_file("cv.pdf", document_type="cv")

    assert result["chunks_processed"] == 2
    assert result["vectors_stored"] == 2
    assert embeddings.received_texts == [
        "I build machine learning systems.",
        "I work with Python.",
    ]
    assert store.received_chunks == chunks
    assert store.received_vectors == vectors


def test_empty_document_skips_embedding_and_storage():
    pipeline, _, embeddings, store = make_pipeline([], [])

    result = pipeline.ingest_file("empty.pdf")

    assert result["chunks_processed"] == 0
    assert result["vectors_stored"] == 0
    assert embeddings.received_texts == []
    assert store.received_chunks == []


def test_pipeline_passes_ingestion_options():
    pipeline, ingestion, _, _ = make_pipeline([], [])

    pipeline.ingest_file(
        "cv.pdf",
        document_type="cv",
        max_words=100,
        overlap=15,
        metadata={"owner": "Eddie"},
    )

    call = ingestion.calls[0]
    assert call["document_type"] == "cv"
    assert call["max_words"] == 100
    assert call["overlap"] == 15
    assert call["metadata"] == {"owner": "Eddie"}


def test_pipeline_rejects_embedding_count_mismatch():
    chunks = [
        make_chunk("chunk_1", "First chunk."),
        make_chunk("chunk_2", "Second chunk."),
    ]

    pipeline, _, _, store = make_pipeline(
        chunks,
        vectors=[[0.1, 0.2]],
    )

    try:
        pipeline.ingest_file("cv.pdf")
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("Expected embedding count mismatch to fail.")

    assert store.received_chunks == []


def test_close_releases_vector_store():
    pipeline, _, _, store = make_pipeline([], [])

    pipeline.close()

    assert store.closed is True
