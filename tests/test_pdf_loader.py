
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
