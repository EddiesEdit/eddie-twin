
from openai import OpenAI


class OpenAIEmbeddings:
    """Generate text embeddings using the OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        client: OpenAI | None = None,
    ) -> None:
        if not api_key.strip() and client is None:
            raise ValueError("An OpenAI API key is required.")

        if not model.strip():
            raise ValueError("Embedding model cannot be empty.")

        if dimensions is not None and dimensions <= 0:
            raise ValueError("Dimensions must be a positive integer.")

        self.model = model
        self.dimensions = dimensions
        self.client = client or OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts while preserving their original order."""
        if not texts:
            return []

        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Text at index {index} must be a non-empty string."
                )

        request = {
            "model": self.model,
            "input": texts,
        }

        if self.dimensions is not None:
            request["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**request)

        # Sort by API-provided input index to preserve input order.
        ordered_data = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in ordered_data]

        if len(vectors) != len(texts):
            raise RuntimeError(
                "The embedding API returned an unexpected number of vectors."
            )

        expected_dimensions = len(vectors[0])

        if any(len(vector) != expected_dimensions for vector in vectors):
            raise RuntimeError("The API returned inconsistent vector dimensions.")

        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query for semantic retrieval."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Query text must be a non-empty string.")

        return self.embed_texts([text])[0]
