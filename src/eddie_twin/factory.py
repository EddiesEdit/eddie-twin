
from dataclasses import dataclass

from eddie_twin.config import Settings, load_settings, validate_api_key
from eddie_twin.embeddings.openai_embeddings import OpenAIEmbeddings
from eddie_twin.generation.generator import Generator
from eddie_twin.rag.pipeline import RAGPipeline
from eddie_twin.rag.query_pipeline import QueryPipeline
from eddie_twin.retrieval.retriever import Retriever
from eddie_twin.vectorstores.qdrant_store import QdrantVectorStore


@dataclass
class EddieTwinApplication:
    """Holds the connected components of the Eddie Twin application."""

    ingestion_pipeline: RAGPipeline
    query_pipeline: QueryPipeline
    embeddings: OpenAIEmbeddings
    vector_store: QdrantVectorStore
    generator: Generator

    def close(self) -> None:
        """Release shared resources through the ingestion pipeline."""
        self.ingestion_pipeline.close()


def create_application(
    settings: Settings | None = None,
    *,
    embeddings: OpenAIEmbeddings | None = None,
    vector_store: QdrantVectorStore | None = None,
    generator: Generator | None = None,
) -> EddieTwinApplication:
    """Build the application while allowing dependencies to be injected."""

    settings = settings or load_settings()

    # Only require credentials when a real OpenAI component must be created.
    if embeddings is None or generator is None:
        validate_api_key(settings)

    if embeddings is None:
        embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )

    if vector_store is None:
        vector_store = QdrantVectorStore(
            path=settings.qdrant_path,
            collection_name=settings.qdrant_collection_name,
            vector_size=1536,
        )

    if generator is None:
        generator = Generator(
            api_key=settings.openai_api_key,
            model=settings.chat_model,
        )

    ingestion_pipeline = RAGPipeline(
        settings=settings,
        embeddings=embeddings,
        vector_store=vector_store,
    )

    retriever = Retriever(
        embeddings=embeddings,
        vector_store=vector_store,
        top_k=settings.top_k,
    )

    query_pipeline = QueryPipeline(
        retriever=retriever,
        generator=generator,
    )

    return EddieTwinApplication(
        ingestion_pipeline=ingestion_pipeline,
        query_pipeline=query_pipeline,
        embeddings=embeddings,
        vector_store=vector_store,
        generator=generator,
    )
