
from types import SimpleNamespace

import pytest

from eddie_twin.generation.generator import Generator


class FakeCompletions:
    def __init__(self, content="I have built machine learning projects."):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content)
                )
            ]
        )


class FakeOpenAIClient:
    def __init__(self, content="I have built machine learning projects."):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(content)
        )


@pytest.fixture
def retrieved_chunks():
    return [
        {
            "id": "point-1",
            "score": 0.95,
            "text": "Built machine learning projects using Python.",
            "metadata": {
                "source": "cv.pdf",
                "page_number": 1,
                "chunk_id": "doc-1:page-1:chunk-0",
            },
        },
        {
            "id": "point-2",
            "score": 0.90,
            "text": "Used Python for data processing.",
            "metadata": {
                "source": "cv.pdf",
                "page_number": 1,
                "chunk_id": "doc-1:page-1:chunk-1",
            },
        },
    ]


def test_generator_returns_answer_and_unique_sources(retrieved_chunks):
    client = FakeOpenAIClient()
    generator = Generator(
        api_key="",
        model="test-model",
        client=client,
    )

    result = generator.generate(
        "What projects have I built?",
        retrieved_chunks,
    )

    assert result["answer"] == "I have built machine learning projects."
    assert result["model"] == "test-model"
    assert result["sources"] == [
        {"source": "cv.pdf", "page_number": 1}
    ]
    assert len(client.chat.completions.calls) == 1


def test_generator_includes_question_and_evidence(retrieved_chunks):
    client = FakeOpenAIClient()
    generator = Generator("", client=client)

    generator.generate("What skills do I have?", retrieved_chunks)

    call = client.chat.completions.calls[0]
    messages = call["messages"]

    assert call["temperature"] == 0.2
    assert call["model"] == "gpt-4o-mini"
    assert messages[0]["role"] == "system"
    assert "untrusted data" in messages[0]["content"]
    assert "What skills do I have?" in messages[1]["content"]
    assert "cv.pdf" in messages[1]["content"]
    assert "Python" in messages[1]["content"]


def test_generator_handles_no_evidence_without_api_call():
    client = FakeOpenAIClient()
    generator = Generator("", client=client)

    result = generator.generate("What is my degree?", [])

    #assert "not have enough information" in result["answer"]
    assert "don't have enough information" in result["answer"]
    assert result["sources"] == []
    assert client.chat.completions.calls == []


def test_generator_rejects_empty_question():
    generator = Generator("", client=FakeOpenAIClient())

    with pytest.raises(ValueError, match="non-empty"):
        generator.generate("  ", [])


def test_generator_rejects_non_list_chunks():
    generator = Generator("", client=FakeOpenAIClient())

    with pytest.raises(TypeError, match="must be a list"):
        generator.generate("What are my skills?", None)


def test_generator_rejects_missing_api_key_without_client():
    with pytest.raises(ValueError, match="API key"):
        Generator("")


def test_generator_rejects_empty_model():
    with pytest.raises(ValueError, match="Model name"):
        Generator("", model=" ", client=FakeOpenAIClient())


def test_generator_rejects_empty_model_response(retrieved_chunks):
    client = FakeOpenAIClient(content=" ")
    generator = Generator("", client=client)

    with pytest.raises(RuntimeError, match="empty answer"):
        generator.generate("What projects have I built?", retrieved_chunks)
