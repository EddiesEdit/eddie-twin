
from pathlib import Path

from eddie_twin.config import Settings
from eddie_twin.embeddings.openai_embeddings import OpenAIEmbeddings
from eddie_twin.factory import create_application
from eddie_twin.generation.generator import Generator
from eddie_twin.vectorstores.qdrant_store import QdrantVectorStore


class FakeEmbeddingClient:
    """Deterministic embedding API substitute for integration tests."""

    class EmbeddingsAPI:
        @staticmethod
        def create(model, input, **kwargs):
            class Item:
                def __init__(self, index, embedding):
                    self.index = index
                    self.embedding = embedding

            class Response:
                pass

            response = Response()
            response.data = [
                Item(index, FakeEmbeddingClient.make_vector(text))
                for index, text in enumerate(input)
            ]
            return response

    def __init__(self):
        self.embeddings = self.EmbeddingsAPI()

    @staticmethod
    def make_vector(text):
        """Create a deterministic 1536-dimensional test vector."""
        vector = [0.0] * 1536
        for char in text.lower():
            vector[ord(char) % 1536] += 1.0

        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude:
            vector = [value / magnitude for value in vector]

        return vector


class FakeGeneratorClient:
    """Avoid external chat API calls during the integration test."""

    class Chat:
        class Completions:
            @staticmethod
            def create(**kwargs):
                class Message:
                    content = "The document says Eddie works in machine learning."

                class Choice:
                    message = Message()

                class Response:
                    choices = [Choice()]

                return Response()

        completions = Completions()

    def __init__(self):
        self.chat = self.Chat()


def create_test_pdf(path: Path):
    """Create a small PDF fixture for the ingestion pipeline."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawString(72, 720, "Eddie works in machine learning.")
    pdf.drawString(72, 700, "Eddie builds retrieval augmented generation systems.")
    pdf.save()


def test_pdf_ingestion_and_retrieval_end_to_end(tmp_path):
    pdf_path = tmp_path / "eddie_cv.pdf"
    create_test_pdf(pdf_path)

    settings = Settings(
        openai_api_key="test-key",
        embedding_model="test-embedding-model",
        chat_model="test-chat-model",
        qdrant_collection_name="integration_test_collection",
        qdrant_path=tmp_path / "qdrant",
        top_k=3,
    )

    embedding_client = FakeEmbeddingClient()
    embeddings = OpenAIEmbeddings(
        api_key="test-key",
        model=settings.embedding_model,
        client=embedding_client,
    )

    generator = Generator(
        api_key="test-key",
        model=settings.chat_model,
        client=FakeGeneratorClient(),
    )

    vector_store = QdrantVectorStore(
        path=settings.qdrant_path,
        collection_name=settings.qdrant_collection_name,
        vector_size=1536,
    )

    app = create_application(
        settings=settings,
        embeddings=embeddings,
        vector_store=vector_store,
        generator=generator,
    )

    try:
        ingestion_result = app.ingestion_pipeline.ingest_file(
            pdf_path,
            document_type="cv",
        )

        assert ingestion_result["chunks_processed"] > 0
        assert ingestion_result["vectors_stored"] == ingestion_result["chunks_processed"]

        results = app.query_pipeline.query(
            "What does Eddie work in?"
        )

        assert results["retrieved_chunks"]
        assert any(
            "machine learning" in chunk["text"].lower()
            for chunk in results["retrieved_chunks"]
        )
        assert "machine learning" in results["answer"].lower()
    finally:
        app.close()
