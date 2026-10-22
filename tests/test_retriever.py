
import pytest

from eddie_twin.retrieval.retriever import Retriever


class FakeEmbeddings:
    def __init__(self):
        self.received_query = None

    def embed_query(self, query):
        self.received_query = query
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self):
        self.received_vector = None
        self.received_top_k = None

    def search(self, query_vector, top_k):
        self.received_vector = query_vector
        self.received_top_k = top_k

        return [
            {
                "id": "point-1",
                "score": 0.95,
                "text": "I built machine learning projects.",
                "metadata": {
                    "source": "cv.pdf",
                    "page_number": 1,
                },
            }
        ]


def test_retriever_returns_relevant_chunks():
    embeddings = FakeEmbeddings()
    store = FakeVectorStore()
    retriever = Retriever(embeddings, store, top_k=5)

    results = retriever.retrieve("What projects have I built?")

    assert embeddings.received_query == "What projects have I built?"
    assert store.received_vector == [0.1, 0.2, 0.3]
    assert store.received_top_k == 5
    assert results[0]["text"] == "I built machine learning projects."
    assert results[0]["metadata"]["source"] == "cv.pdf"


def test_retriever_allows_custom_top_k():
    store = FakeVectorStore()
    retriever = Retriever(FakeEmbeddings(), store, top_k=5)

    retriever.retrieve("Tell me about my experience", top_k=2)

    assert store.received_top_k == 2


def test_retriever_rejects_empty_query():
    retriever = Retriever(FakeEmbeddings(), FakeVectorStore())

    with pytest.raises(ValueError, match="non-empty"):
        retriever.retrieve("   ")


def test_retriever_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        Retriever(FakeEmbeddings(), FakeVectorStore(), top_k=0)


def test_retriever_rejects_invalid_query_top_k():
    retriever = Retriever(FakeEmbeddings(), FakeVectorStore())

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("What skills do I have?", top_k=0)
