
import pytest

from eddie_twin.rag.query_pipeline import QueryPipeline


class FakeRetriever:
    def __init__(self):
        self.received_question = None
        self.received_top_k = None

    def retrieve(self, query, top_k=None):
        self.received_question = query
        self.received_top_k = top_k

        return [
            {
                "id": "point-1",
                "score": 0.95,
                "text": "Built machine learning projects using Python.",
                "metadata": {
                    "source": "cv.pdf",
                    "page_number": 1,
                },
            }
        ]


class FakeGenerator:
    def __init__(self):
        self.received_question = None
        self.received_chunks = None

    def generate(self, question, retrieved_chunks):
        self.received_question = question
        self.received_chunks = retrieved_chunks

        return {
            "answer": "I have built machine learning projects using Python.",
            "sources": [
                {"source": "cv.pdf", "page_number": 1}
            ],
            "model": "test-model",
        }


def test_query_pipeline_connects_retriever_and_generator():
    retriever = FakeRetriever()
    generator = FakeGenerator()
    pipeline = QueryPipeline(retriever, generator)

    result = pipeline.query("What projects have I built?")

    assert retriever.received_question == "What projects have I built?"
    assert generator.received_question == "What projects have I built?"
    assert generator.received_chunks == result["retrieved_chunks"]
    assert result["answer"] == (
        "I have built machine learning projects using Python."
    )
    assert result["sources"] == [
        {"source": "cv.pdf", "page_number": 1}
    ]
    assert result["model"] == "test-model"


def test_query_pipeline_passes_custom_top_k():
    retriever = FakeRetriever()
    pipeline = QueryPipeline(retriever, FakeGenerator())

    pipeline.query("What are my skills?", top_k=3)

    assert retriever.received_top_k == 3


def test_query_pipeline_defaults_top_k_to_none():
    retriever = FakeRetriever()
    pipeline = QueryPipeline(retriever, FakeGenerator())

    pipeline.query("Tell me about my experience.")

    assert retriever.received_top_k is None


def test_query_pipeline_rejects_empty_question():
    pipeline = QueryPipeline(FakeRetriever(), FakeGenerator())

    with pytest.raises(ValueError, match="non-empty"):
        pipeline.query("   ")


def test_query_pipeline_returns_retrieved_chunks():
    pipeline = QueryPipeline(FakeRetriever(), FakeGenerator())

    result = pipeline.query("What skills do I have?")

    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0]["metadata"]["source"] == "cv.pdf"
