
from typing import Any

from eddie_twin.embeddings.openai_embeddings import OpenAIEmbeddings
from eddie_twin.vectorstores.qdrant_store import QdrantVectorStore


class Retriever:
    """Retrieve relevant document chunks for a user query."""

    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        vector_store: QdrantVectorStore,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        self.embeddings = embeddings
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most relevant chunks for a question."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        result_limit = self.top_k if top_k is None else top_k

        if result_limit <= 0:
            raise ValueError("top_k must be positive.")

        # 1. Convert the user's question into an embedding.
        query_vector = self.embeddings.embed_query(query)

        # 2. Search Qdrant for similar vectors.
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=result_limit,
        )

        # 3. Return results in relevance order, preserving provenance.
        return results
