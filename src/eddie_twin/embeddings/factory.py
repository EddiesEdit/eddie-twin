
from eddie_twin.config import load_settings, validate_api_key
from eddie_twin.embeddings.openai_embeddings import OpenAIEmbeddings


def create_embeddings() -> OpenAIEmbeddings:
    """Build the embedding client from application settings."""
    settings = load_settings()
    validate_api_key(settings)

    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
