
import re
from typing import Any


class OutputGuard:
    """Apply basic safety checks to generated responses."""

    SECRET_PATTERNS = {
        "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "private_key": re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        ),
    }

    def __init__(self, max_length: int = 20_000) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        self.max_length = max_length

    def validate(self, response: str) -> dict[str, Any]:
        """Return a structured assessment of a generated response."""

        if not isinstance(response, str):
            raise ValueError("Response must be a string.")

        if not response.strip():
            return {
                "is_safe": False,
                "reasons": ["empty_response"],
            }

        if len(response) > self.max_length:
            return {
                "is_safe": False,
                "reasons": ["response_too_long"],
            }

        findings = [
            name
            for name, pattern in self.SECRET_PATTERNS.items()
            if pattern.search(response)
        ]

        if findings:
            return {
                "is_safe": False,
                "reasons": ["possible_secret_exposure"],
                "findings": findings,
            }

        return {
            "is_safe": True,
            "reasons": [],
            "findings": [],
        }