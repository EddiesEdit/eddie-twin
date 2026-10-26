
import pytest

from eddie_twin.preprocessing.chunker import chunk_page
from eddie_twin.preprocessing.text_cleaner import clean_text


def test_clean_text_normalizes_whitespace():
    text = "  Hello   world.  \r\n\r\n\r\nPython  "

    assert clean_text(text) == "Hello world.\n\nPython"


def test_chunk_page_preserves_metadata():
    text = " ".join(f"word{i}" for i in range(250))

    chunks = chunk_page(
        text=text,
        document_id="doc_example",
        source="cv.pdf",
        page_number=2,
        max_words=100,
        overlap=20,
    )

    assert len(chunks) == 3
    assert chunks[0]["document_id"] == "doc_example"
    assert chunks[0]["source"] == "cv.pdf"
    assert chunks[0]["page_number"] == 2
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["word_count"] == 100
    assert chunks[1]["text"].split()[0] == "word80"
    assert len(chunks[0]["content_hash"]) == 64
    assert chunks[0]["document_type"] == "unknown"
    assert chunks[0]["ingested_at"]


def test_empty_text_returns_no_chunks():
    assert chunk_page(
        text="",
        document_id="doc_example",
        source="cv.pdf",
        page_number=1,
    ) == []


def test_invalid_chunk_settings_raise_error():
    with pytest.raises(ValueError):
        chunk_page(
            "Some text", "doc_example", "cv.pdf", 1, max_words=0
        )

    with pytest.raises(ValueError):
        chunk_page(
            "Some text", "doc_example", "cv.pdf", 1,
            max_words=10, overlap=10,
        )


def test_invalid_page_number_raises_error():
    with pytest.raises(ValueError, match="page_number"):
        chunk_page(
            text="Some text",
            document_id="doc_example",
            source="cv.pdf",
            page_number=0,
        )


def test_chunks_never_exceed_max_words():
    text = " ".join(f"word{i}" for i in range(250))

    chunks = chunk_page(
        text=text,
        document_id="doc_example",
        source="cv.pdf",
        page_number=1,
        max_words=100,
        overlap=20,
    )

    assert all(len(chunk["text"].split()) <= 100 for chunk in chunks)


def test_all_words_are_preserved_in_order():
    words = [f"word{i}" for i in range(250)]

    chunks = chunk_page(
        text=" ".join(words),
        document_id="doc_example",
        source="cv.pdf",
        page_number=1,
        max_words=100,
        overlap=20,
    )

    step = 80

    for index, chunk in enumerate(chunks):
        start = index * step
        expected_chunk = words[start : start + 100]

        assert chunk["text"].split() == expected_chunk

    # Confirm every original word is covered by the chunks.
    covered_words = set()

    for chunk in chunks:
        covered_words.update(chunk["text"].split())

    assert covered_words == set(words)