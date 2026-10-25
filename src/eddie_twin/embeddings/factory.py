
from eddie_twin.config import Settings, load_settings, validate_api_key
from eddie_twin.embeddings.openai_embeddings import OpenAIEmbeddings


def create_embeddings(
    settings: Settings | None = None,
) -> OpenAIEmbeddings:
    """Create the configured OpenAI embedding client."""

    settings = settings or load_settings()
    validate_api_key(settings)

    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
