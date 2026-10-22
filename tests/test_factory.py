
import pytest

from eddie_twin.config import Settings
from eddie_twin.factory import create_application


class FakeEmbeddings:
    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self):
        self.closed = False
        self.search_calls = []

    def search(self, query_vector, top_k=5):
        self.search_calls.append((query_vector, top_k))
        return [
            {
                "id": "chunk-1",
                "score": 0.95,
                "text": "Eddie works in machine learning.",
                "metadata": {
                    "source": "cv.pdf",
                    "page_number": 1,
                },
            }
        ]

    def close(self):
        self.closed = True


class FakeGenerator:
    model = "fake-model"

    def generate(self, question, retrieved_chunks):
        return {
            "answer": "The document says Eddie works in machine learning.",
            "sources": [{"source": "cv.pdf", "page_number": 1}],
            "model": self.model,
        }


@pytest.fixture
def settings(tmp_path):
    return Settings(
        openai_api_key="",
        embedding_model="test-embedding-model",
        chat_model="test-chat-model",
        qdrant_collection_name="test_collection",
        qdrant_path=tmp_path / "qdrant",
        top_k=3,
    )


def test_factory_wires_shared_dependencies(settings):
    embeddings = FakeEmbeddings()
    vector_store = FakeVectorStore()
    generator = FakeGenerator()

    app = create_application(
        settings=settings,
        embeddings=embeddings,
        vector_store=vector_store,
        generator=generator,
    )

    assert app.embeddings is embeddings
    assert app.vector_store is vector_store
    assert app.generator is generator

    assert app.ingestion_pipeline.embeddings is embeddings
    assert app.ingestion_pipeline.vector_store is vector_store

    assert app.query_pipeline.retriever.embeddings is embeddings
    assert app.query_pipeline.retriever.vector_store is vector_store
    assert app.query_pipeline.retriever.top_k == 3
    assert app.query_pipeline.generator is generator


def test_factory_query_pipeline_works_without_api_calls(settings):
    vector_store = FakeVectorStore()

    app = create_application(
        settings=settings,
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
        generator=FakeGenerator(),
    )

    result = app.query_pipeline.query("What does Eddie do?")

    assert "machine learning" in result["answer"]
    assert result["model"] == "fake-model"
    assert len(result["retrieved_chunks"]) == 1
    assert vector_store.search_calls == [([0.1, 0.2, 0.3], 3)]


def test_factory_closes_shared_vector_store(settings):
    vector_store = FakeVectorStore()

    app = create_application(
        settings=settings,
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
        generator=FakeGenerator(),
    )

    app.close()

    assert vector_store.closed is True


def test_factory_requires_api_key_for_real_components(settings):
    with pytest.raises(
        ValueError,
        match="OPENAI_API_KEY is missing",
    ):
        create_application(settings=settings)
