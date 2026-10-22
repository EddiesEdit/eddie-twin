
from types import SimpleNamespace

import pytest

from eddie_twin.embeddings.openai_embeddings import OpenAIEmbeddings


class FakeEmbeddingClient:
    """Simulate the OpenAI embeddings API without network requests."""

    def __init__(self):
        self.embeddings = self
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs

        texts = kwargs["input"]

        # Deliberately return results in reverse order to test sorting.
        data = [
            SimpleNamespace(index=i, embedding=[float(i), 0.5, 1.0])
            for i in reversed(range(len(texts)))
        ]

        return SimpleNamespace(data=data)


def test_embed_texts_returns_vectors_in_input_order():
    client = FakeEmbeddingClient()
    embeddings = OpenAIEmbeddings(
        api_key="test-key",
        client=client,
    )

    vectors = embeddings.embed_texts(["first text", "second text"])

    assert vectors == [[0.0, 0.5, 1.0], [1.0, 0.5, 1.0]]


def test_embed_texts_returns_empty_list_for_empty_input():
    embeddings = OpenAIEmbeddings(
        api_key="test-key",
        client=FakeEmbeddingClient(),
    )

    assert embeddings.embed_texts([]) == []


def test_embed_texts_rejects_blank_input():
    embeddings = OpenAIEmbeddings(
        api_key="test-key",
        client=FakeEmbeddingClient(),
    )

    with pytest.raises(ValueError, match="non-empty"):
        embeddings.embed_texts(["valid text", "  "])


def test_embed_query_returns_one_vector():
    embeddings = OpenAIEmbeddings(
        api_key="test-key",
        client=FakeEmbeddingClient(),
    )

    assert embeddings.embed_query("Who am I?") == [0.0, 0.5, 1.0]


def test_dimensions_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        OpenAIEmbeddings(api_key="test-key", dimensions=0)


def test_configured_dimensions_are_sent_to_api():
    client = FakeEmbeddingClient()
    embeddings = OpenAIEmbeddings(
        api_key="test-key",
        dimensions=3,
        client=client,
    )

    embeddings.embed_texts(["hello"])

    assert client.last_request["dimensions"] == 3
