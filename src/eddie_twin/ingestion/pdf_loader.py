
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader


class DocumentPage(TypedDict):
    source: str
    page_number: int
    text: str


class PDFLoader:
    """Extract text from PDFs with basic resource and format limits."""

    def __init__(
        self,
        max_file_size_bytes: int = 20 * 1024 * 1024,
        max_pages: int = 500,
        max_extracted_chars: int = 5_000_000,
    ) -> None:
        if max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be positive.")
        if max_pages <= 0:
            raise ValueError("max_pages must be positive.")
        if max_extracted_chars <= 0:
            raise ValueError("max_extracted_chars must be positive.")

        self.max_file_size_bytes = max_file_size_bytes
        self.max_pages = max_pages
        self.max_extracted_chars = max_extracted_chars

    def load(self, file_path: str | Path) -> list[DocumentPage]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a PDF file, got: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.name}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise ValueError("PDF file is empty.")

        if file_size > self.max_file_size_bytes:
            raise ValueError(
                f"PDF exceeds the maximum file size of "
                f"{self.max_file_size_bytes} bytes."
            )

        # A PDF header is normally near the beginning of the file.
        try:
            with path.open("rb") as file:
                header = file.read(1024)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read PDF '{path.name}'."
            ) from exc

        if b"%PDF-" not in header:
            raise ValueError("File does not have a valid PDF signature.")

        try:
            reader = PdfReader(path, strict=True)

            if reader.is_encrypted:
                raise ValueError(
                    f"Encrypted PDFs are not supported: {path.name}"
                )

            page_count = len(reader.pages)
            if page_count > self.max_pages:
                raise ValueError(
                    f"PDF has {page_count} pages; maximum allowed is "
                    f"{self.max_pages}."
                )

            pages: list[DocumentPage] = []
            total_chars = 0

            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                total_chars += len(text)

                if total_chars > self.max_extracted_chars:
                    raise ValueError(
                        "PDF exceeds the maximum extracted-text limit."
                    )

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
            # Avoid exposing internal parser details to the caller.
            raise RuntimeError(
                f"Failed to read PDF '{path.name}'."
            ) from exc
