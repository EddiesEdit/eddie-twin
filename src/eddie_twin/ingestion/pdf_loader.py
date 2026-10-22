
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader


class DocumentPage(TypedDict):
    source: str
    page_number: int
    text: str


class PDFLoader:
    """Extract text and metadata from PDF documents."""

    def load(self, file_path: str | Path) -> list[DocumentPage]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a PDF file, got: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.name}")

        try:
            reader = PdfReader(path)

            if reader.is_encrypted:
                raise ValueError(
                    f"Encrypted PDFs are not supported: {path.name}"
                )

            pages: list[DocumentPage] = []

            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()

                if not text:
                    continue

                pages.append(
                    {
                        "source": path.name,
                        "page_number": page_number,
                        "text": text,
                    }
                )

            return pages

        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read PDF '{path.name}': {exc}"
            ) from exc
