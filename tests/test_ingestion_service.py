
from pathlib import Path

from eddie_twin.ingestion.service import IngestionService


class FakePDFLoader:
    """Return predictable pages without requiring a real PDF."""

    def load(self, file_path: str | Path) -> list[dict]:
        return [
            {
                "source": Path(file_path).name,
                "page_number": 1,
                "text": " ".join(f"word{i}" for i in range(25)),
            },
            {
                "source": Path(file_path).name,
                "page_number": 2,
                "text": "This is the second page.",
            },
        ]


def test_ingestion_service_returns_document_and_chunks(tmp_path: Path):
    pdf_path = tmp_path / "cv.pdf"
    pdf_path.write_bytes(b"test PDF content")

    service = IngestionService(pdf_loader=FakePDFLoader())

    result = service.ingest_pdf(
        pdf_path,
        document_type="cv",
        max_words=10,
        overlap=2,
    )

    assert "document" in result
    assert "chunks" in result
    assert result["document"]["source"] == "cv.pdf"
    assert result["document"]["document_type"] == "cv"
    assert result["document"]["content_hash"]
    assert result["chunks"]
    assert all(
        chunk["document_id"] == result["document"]["document_id"]
        for chunk in result["chunks"]
    )


def test_ingestion_service_preserves_page_numbers(tmp_path: Path):
    pdf_path = tmp_path / "cv.pdf"
    pdf_path.write_bytes(b"test PDF content")

    service = IngestionService(pdf_loader=FakePDFLoader())

    result = service.ingest_pdf(pdf_path, max_words=10, overlap=2)

    page_numbers = {chunk["page_number"] for chunk in result["chunks"]}

    assert page_numbers == {1, 2}


def test_same_file_contents_produce_same_document_id(tmp_path: Path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "renamed.pdf"

    first.write_bytes(b"identical content")
    second.write_bytes(b"identical content")

    service = IngestionService(pdf_loader=FakePDFLoader())

    first_result = service.ingest_pdf(first)
    second_result = service.ingest_pdf(second)

    assert (
        first_result["document"]["document_id"]
        == second_result["document"]["document_id"]
    )
