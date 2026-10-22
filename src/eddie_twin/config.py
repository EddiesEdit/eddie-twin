
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    embedding_model: str
    chat_model: str
    qdrant_collection_name: str
    qdrant_path: Path
    top_k: int


def load_settings() -> Settings:
    """Load and validate application configuration."""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    embedding_model = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    ).strip()

    chat_model = os.getenv(
        "OPENAI_CHAT_MODEL",
        "gpt-4o-mini",
    ).strip()

    collection_name = os.getenv(
        "QDRANT_COLLECTION_NAME",
        "eddie_twin_documents",
    ).strip()

    qdrant_path_value = os.getenv(
        "QDRANT_PATH",
        "storage/qdrant",
    ).strip()

    try:
        top_k = int(os.getenv("TOP_K", "5"))
    except ValueError as exc:
        raise ValueError("TOP_K must be an integer.") from exc

    if not embedding_model:
        raise ValueError("Embedding model cannot be empty.")

    if not chat_model:
        raise ValueError("Chat model cannot be empty.")

    if not collection_name:
        raise ValueError("Qdrant collection name cannot be empty.")

    if not qdrant_path_value:
        raise ValueError("QDRANT_PATH cannot be empty.")

    if top_k < 1:
        raise ValueError("TOP_K must be at least 1.")

    qdrant_path = Path(qdrant_path_value)

    if not qdrant_path.is_absolute():
        qdrant_path = PROJECT_ROOT / qdrant_path

    return Settings(
        openai_api_key=api_key,
        embedding_model=embedding_model,
        chat_model=chat_model,
        qdrant_collection_name=collection_name,
        qdrant_path=qdrant_path,
        top_k=top_k,
    )


def validate_api_key(settings: Settings) -> None:
    """Require an API key when a feature needs OpenAI."""

    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. "
            "Add it to your local .env file."
        )
