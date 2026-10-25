
from typing import Any

from eddie_twin.security.input_guard import InputGuard
from eddie_twin.security.prompt_injection import PromptInjectionDetector


class SecurityDefense:
    """Coordinate input validation and injection-risk assessment."""

    def __init__(
        self,
        max_input_length: int = 8_000,
        injection_threshold: float = 0.80,
    ) -> None:
        if not 0.0 <= injection_threshold <= 1.0:
            raise ValueError(
                "injection_threshold must be between 0 and 1."
            )

        self.input_guard = InputGuard(max_length=max_input_length)
        self.detector = PromptInjectionDetector()
        self.injection_threshold = injection_threshold

    def inspect_input(self, user_input: str) -> dict[str, Any]:
        """Validate input and assess its injection risk."""

        clean_input = self.input_guard.validate(user_input)
        detection = self.detector.detect(clean_input)

        return {
            "clean_input": clean_input,
            "detection": detection,
            "should_block": (
                detection["is_injection"]
                and detection["confidence"] >= self.injection_threshold
            ),
        }
