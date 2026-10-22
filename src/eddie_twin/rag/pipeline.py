
from pathlib import Path
from typing import Any

from eddie_twin.config import Settings, load_settings, validate_api_key
from eddie_twin.embeddings.openai_embeddings import OpenAIEmbeddings
from eddie_twin.ingestion.service import IngestionService
from eddie_twin.vectorstores.qdrant_store import QdrantVectorStore


class RAGPipeline:
    """Coordinate document ingestion, embedding, and vector storage."""

    def __init__(
        self,
        settings: Settings | None = None,
        ingestion_service: IngestionService | None = None,
        embeddings: OpenAIEmbeddings | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self.settings = settings or load_settings()

        self.ingestion_service = ingestion_service or IngestionService()

        if embeddings is None:
            validate_api_key(self.settings)
            embeddings = OpenAIEmbeddings(
                api_key=self.settings.openai_api_key,
                model=self.settings.embedding_model,
            )

        self.embeddings = embeddings

        self.vector_store = vector_store or QdrantVectorStore(
            path=self.settings.qdrant_path,
            collection_name=self.settings.qdrant_collection_name,
            vector_size=1536,
        )

    def ingest_file(
        self,
        file_path: str | Path,
        document_type: str = "unknown",
        max_words: int = 200,
        overlap: int = 30,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingest a document and store its chunks as vectors."""

        # Stage 1: Load, identify, and chunk the document.
        ingestion_result = self.ingestion_service.ingest_pdf(
            file_path=file_path,
            document_type=document_type,
            max_words=max_words,
            overlap=overlap,
            metadata=metadata,
        )

        document = ingestion_result["document"]
        chunks = ingestion_result["chunks"]

        # Empty or textless PDFs should not trigger embedding requests.
        if not chunks:
            return {
                "document": document,
                "chunks_processed": 0,
                "vectors_stored": 0,
            }

        # Stage 2: Generate one embedding for each chunk.
        texts = [chunk["text"] for chunk in chunks]
        vectors = self.embeddings.embed_texts(texts)

        if len(vectors) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match the number of chunks."
            )

        # Stage 3: Store vectors alongside chunk text and metadata.
        stored_count = self.vector_store.upsert_chunks(chunks, vectors)

        return {
            "document": document,
            "chunks_processed": len(chunks),
            "vectors_stored": stored_count,
        }

    def close(self) -> None:
        """Release vector-store resources."""
        self.vector_store.close()
