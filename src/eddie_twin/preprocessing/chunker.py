
import hashlib

from eddie_twin.metadata import ChunkMetadata
from eddie_twin.preprocessing.text_cleaner import clean_text


def chunk_page(
    text: str,
    document_id: str,
    source: str,
    page_number: int,
    document_type: str = "unknown",
    max_words: int = 200,
    overlap: int = 30,
) -> list[dict]:
    """Split page text into chunks with validated metadata."""

    if max_words <= 0:
        raise ValueError("max_words must be positive")

    if overlap < 0 or overlap >= max_words:
        raise ValueError("overlap must be >= 0 and < max_words")

    if page_number < 1:
        raise ValueError("page_number must be at least 1")

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    words = cleaned_text.split()
    chunks = []
    step = max_words - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]

        if not chunk_words:
            break

        chunk_text = " ".join(chunk_words)
        chunk_index = len(chunks)

        content_hash = hashlib.sha256(
            chunk_text.encode("utf-8")
        ).hexdigest()

        metadata = ChunkMetadata(
            chunk_id=(
                f"{document_id}:page-{page_number}:chunk-{chunk_index}"
            ),
            document_id=document_id,
            source=source,
            document_type=document_type,
            page_number=page_number,
            chunk_index=chunk_index,
            word_count=len(chunk_words),
            char_count=len(chunk_text),
            content_hash=content_hash,
        )

        chunks.append(
            {
                **metadata.model_dump(mode="json"),
                "text": chunk_text,
            }
        )

        if start + max_words >= len(words):
            break

    return chunks
