
import pytest

from eddie_twin.security.output_guard import OutputGuard


def test_accepts_normal_response():
    result = OutputGuard().validate("Eddie works in machine learning.")

    assert result["is_safe"] is True


def test_rejects_empty_response():
    result = OutputGuard().validate(" ")

    assert result["is_safe"] is False
    assert "empty_response" in result["reasons"]


def test_rejects_response_over_limit():
    result = OutputGuard(max_length=5).validate("too long")

    assert result["is_safe"] is False
    assert "response_too_long" in result["reasons"]


def test_detects_possible_api_key_exposure():
    response = "Example key: sk-" + ("A" * 25)

    result = OutputGuard().validate(response)

    assert result["is_safe"] is False
    assert "possible_secret_exposure" in result["reasons"]


def test_rejects_non_string_response():
    with pytest.raises(ValueError, match="string"):
        OutputGuard().validate(None)
