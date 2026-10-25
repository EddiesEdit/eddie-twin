
from pathlib import Path

import pytest

from eddie_twin.ingestion.pdf_loader import PDFLoader


def test_missing_pdf_raises_error(tmp_path: Path):
    loader = PDFLoader()

    with pytest.raises(FileNotFoundError):
        loader.load(tmp_path / "missing.pdf")


def test_non_pdf_file_raises_error(tmp_path: Path):
    file_path = tmp_path / "document.txt"
    file_path.write_text("Sample text", encoding="utf-8")

    loader = PDFLoader()

    with pytest.raises(ValueError, match="Expected a PDF file"):
        loader.load(file_path)


def test_rejects_file_with_pdf_extension_but_no_signature(tmp_path: Path):
    file_path = tmp_path / "fake.pdf"
    file_path.write_text("This is not a PDF.", encoding="utf-8")

    with pytest.raises(ValueError, match="PDF signature"):
        PDFLoader().load(file_path)


def test_rejects_empty_pdf_file(tmp_path: Path):
    file_path = tmp_path / "empty.pdf"
    file_path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        PDFLoader().load(file_path)


def test_rejects_pdf_over_size_limit(tmp_path: Path):
    file_path = tmp_path / "large.pdf"
    file_path.write_bytes(b"%PDF-" + b"x" * 100)

    loader = PDFLoader(max_file_size_bytes=50)

    with pytest.raises(ValueError, match="maximum file size"):
        loader.load(file_path)


def test_rejects_invalid_loader_limits():
    with pytest.raises(ValueError, match="max_file_size_bytes"):
        PDFLoader(max_file_size_bytes=0)

    with pytest.raises(ValueError, match="max_pages"):
        PDFLoader(max_pages=0)

    with pytest.raises(ValueError, match="max_extracted_chars"):
        PDFLoader(max_extracted_chars=0)
