
from unittest.mock import patch

import pytest

from eddie_twin.config import load_settings
from eddie_twin.embeddings.factory import create_embeddings


def test_factory_uses_configured_embedding_model():
    with (
        patch("eddie_twin.embeddings.factory.load_settings") as mock_settings,
        patch("eddie_twin.embeddings.factory.validate_api_key"),
        patch("eddie_twin.embeddings.factory.OpenAIEmbeddings") as mock_class,
    ):
        settings = load_settings()
        mock_settings.return_value = settings

        create_embeddings()

        mock_class.assert_called_once_with(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )


def test_factory_rejects_missing_api_key():
    with (
        patch("eddie_twin.embeddings.factory.load_settings") as mock_settings,
        patch(
            "eddie_twin.embeddings.factory.validate_api_key",
            side_effect=ValueError("OPENAI_API_KEY is missing."),
        ),
        patch("eddie_twin.embeddings.factory.OpenAIEmbeddings") as mock_class,
    ):
        mock_settings.return_value = load_settings()

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_embeddings()

        mock_class.assert_not_called()
