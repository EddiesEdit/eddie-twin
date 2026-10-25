
from typing import Any

from eddie_twin.generation.generator import Generator
from eddie_twin.retrieval.retriever import Retriever
from eddie_twin.security.audit import SecurityAudit
from eddie_twin.security.defense import SecurityDefense
from eddie_twin.security.output_guard import OutputGuard


class QueryPipeline:
    """Run RAG queries through security checks."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        defense: SecurityDefense | None = None,
        output_guard: OutputGuard | None = None,
        audit: SecurityAudit | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.defense = defense or SecurityDefense()
        self.output_guard = output_guard or OutputGuard()
        self.audit = audit or SecurityAudit()

    def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Validate, retrieve, generate, and validate the response."""

        inspection = self.defense.inspect_input(question)

        if inspection["should_block"]:
            categories = inspection["detection"]["categories"]

            self.audit.record(
                "query_blocked",
                ",".join(categories) or "injection_risk",
                blocked=True,
            )

            return {
                "answer": (
                    "I couldn't process that message because it triggered "
                    "a security check. Please rephrase your request."
                ),
                "sources": [],
                "retrieved_chunks": [],
                "security": {
                    "blocked": True,
                    "reason": "injection_risk",
                },
            }

        clean_question = inspection["clean_input"]

        retrieved_chunks = self.retriever.retrieve(
            query=clean_question,
            top_k=top_k,
        )

        result = self.generator.generate(
            question=clean_question,
            retrieved_chunks=retrieved_chunks,
        )

        output_check = self.output_guard.validate(result["answer"])

        if not output_check["is_safe"]:
            self.audit.record(
                "output_blocked",
                ",".join(output_check["reasons"]),
                blocked=True,
            )

            return {
                "answer": (
                    "I couldn't return the generated response because "
                    "it did not pass a security check."
                ),
                "sources": [],
                "retrieved_chunks": [],
                "security": {
                    "blocked": True,
                    "reason": "unsafe_output",
                },
            }

        self.audit.record(
            "query_completed",
            "passed",
            blocked=False,
        )

        return {
            **result,
            "retrieved_chunks": retrieved_chunks,
            "security": {
                "blocked": False,
                "reason": None,
            },
        }
