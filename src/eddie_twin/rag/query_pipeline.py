
from typing import Any

from eddie_twin.generation.generator import Generator
from eddie_twin.retrieval.retriever import Retriever


class QueryPipeline:
    """Orchestrate retrieval and grounded answer generation."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve evidence and generate an answer."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        # Step 1: Retrieve relevant document chunks.
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        # Step 2: Generate an answer grounded in the retrieved evidence.
        result = self.generator.generate(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        # Step 3: Return the answer and retrieval details.
        return {
            **result,
            "retrieved_chunks": retrieved_chunks,
        }
