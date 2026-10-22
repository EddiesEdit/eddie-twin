
import hashlib
from pathlib import Path


def calculate_file_hash(file_path: str | Path) -> str:
    """Return a SHA-256 hash of the original file."""

    path = Path(file_path)
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def create_document_id(file_hash: str) -> str:
    """Create a content-based identity for a document."""

    if len(file_hash) != 64:
        raise ValueError("Expected a SHA-256 hash.")

    try:
        int(file_hash, 16)
    except ValueError as exc:
        raise ValueError("Hash must contain hexadecimal characters.") from exc

    return f"doc_{file_hash}"
