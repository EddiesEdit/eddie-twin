
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models


class QdrantVectorStore:
    """Persist and retrieve document chunks using Qdrant."""

    def __init__(
        self,
        path: str | Path,
        collection_name: str,
        vector_size: int = 1536,
        client: QdrantClient | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        if vector_size <= 0:
            raise ValueError("Vector size must be positive.")

        self.collection_name = collection_name
        self.vector_size = vector_size

        # An injected client is useful for testing and alternate deployments.
        self.client = client or QdrantClient(path=str(path))

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

        # Fail early if an existing collection has incompatible dimensions.
        info = self.client.get_collection(self.collection_name)
        config = info.config.params.vectors

        if isinstance(config, dict):
            raise ValueError(
                "This store expects a single-vector Qdrant collection."
            )

        if config.size != self.vector_size:
            raise ValueError(
                f"Collection vector size is {config.size}, "
                f"but this store expects {self.vector_size}."
            )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Convert a chunk identifier into a deterministic UUID."""
        if not chunk_id.strip():
            raise ValueError("chunk_id cannot be empty.")

        return str(uuid5(NAMESPACE_URL, chunk_id))

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> int:
        """Store chunks and their embeddings; return the number processed."""
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one vector.")

        if not chunks:
            return 0

        points = []

        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.vector_size:
                raise ValueError(
                    f"Expected vector size {self.vector_size}, "
                    f"received {len(vector)}."
                )

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("Each chunk must have a valid chunk_id.")

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Each chunk must contain non-empty text.")

            payload = {key: value for key, value in chunk.items()}

            points.append(
                models.PointStruct(
                    id=self._point_id(chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the most similar chunks for a query embedding."""
        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Expected query vector size {self.vector_size}, "
                f"received {len(query_vector)}."
            )

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                {
                    "id": str(point.id),
                    "score": point.score,
                    "text": payload.get("text", ""),
                    "metadata": {
                        key: value
                        for key, value in payload.items()
                        if key != "text"
                    },
                }
            )

        return results

    def close(self) -> None:
        """Release the Qdrant client resources."""
        self.client.close()
