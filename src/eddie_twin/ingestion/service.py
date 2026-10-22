
from pathlib import Path
from typing import Any

from eddie_twin.ingestion.document_identity import (
    calculate_file_hash,
    create_document_id,
)
from eddie_twin.ingestion.pdf_loader import PDFLoader
from eddie_twin.metadata import DocumentMetadata
from eddie_twin.preprocessing.chunker import chunk_page


class IngestionService:
    """Coordinate document loading, identity, chunking, and metadata."""

    def __init__(self, pdf_loader: PDFLoader | None = None) -> None:
        self.pdf_loader = pdf_loader or PDFLoader()

    def ingest_pdf(
        self,
        file_path: str | Path,
        document_type: str = "unknown",
        max_words: int = 200,
        overlap: int = 30,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)

        # 1. Extract text from the PDF.
        pages = self.pdf_loader.load(path)

        # 2. Generate a stable identity from the file contents.
        content_hash = calculate_file_hash(path)
        document_id = create_document_id(content_hash)

        # 3. Create document-level metadata.
        document = DocumentMetadata(
            document_id=document_id,
            source=path.name,
            document_type=document_type,
            content_hash=content_hash,
            metadata=metadata or {},
        )

        # 4. Chunk every extracted page while preserving page numbers.
        chunks: list[dict[str, Any]] = []

        for page in pages:
            page_chunks = chunk_page(
                text=page["text"],
                document_id=document_id,
                source=page["source"],
                page_number=page["page_number"],
                document_type=document_type,
                max_words=max_words,
                overlap=overlap,
            )
            chunks.extend(page_chunks)

        # 5. Return JSON-friendly data for later pipeline stages.
        return {
            "document": document.model_dump(mode="json"),
            "chunks": chunks,
        }
