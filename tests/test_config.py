
from eddie_twin.config import load_settings


def test_default_settings():
    settings = load_settings()

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chat_model == "gpt-4o-mini"
    assert settings.qdrant_collection_name == "eddie_twin_documents"
    assert settings.qdrant_path.is_absolute()
    assert settings.top_k == 5
