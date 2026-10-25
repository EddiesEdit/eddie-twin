
import re
from typing import Any


class PromptInjectionDetector:
    """Rule-based detector for common prompt-injection patterns.

    This is a risk signal, not a complete security boundary.
    """

    PATTERNS: dict[str, tuple[str, float]] = {
        "instruction_override": (
            r"\bignore\s+(all\s+|any\s+|the\s+)?"
            r"(previous|prior|above|preceding)\s+instructions\b",
            0.85,
        ),
        "system_prompt_extraction": (
            r"\b(reveal|show|print|display|repeat|output)\b"
            r".{0,80}\b(system prompt|hidden instructions|"
            r"developer message)\b",
            0.85,
        ),
        "role_override": (
            r"\b(you are now|act as if you are|"
            r"pretend your rules do not apply)\b",
            0.70,
        ),
        "safety_bypass": (
            r"\b(bypass|disable|override|circumvent)\b"
            r".{0,60}\b(safety|security|guardrails|restrictions)\b",
            0.80,
        ),
        "untrusted_instruction_execution": (
            r"\b(follow|execute|obey)\b.{0,60}"
            r"\b(instructions|commands)\b.{0,60}"
            r"\b(document|webpage|retrieved text|file)\b",
            0.70,
        ),
    }

    def detect(self, text: str) -> dict[str, Any]:
        """Assess a message and return risk signals."""

        if not isinstance(text, str):
            raise ValueError("Text must be a string.")

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        findings: list[dict[str, Any]] = []

        for category, (pattern, confidence) in self.PATTERNS.items():
            if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                findings.append(
                    {
                        "category": category,
                        "confidence": confidence,
                    }
                )

        # Multiple distinct signals increase the risk score.
        # This is a heuristic score, not a calibrated probability.
        if findings:
            confidence = min(
                0.99,
                max(item["confidence"] for item in findings)
                + 0.05 * (len(findings) - 1),
            )
        else:
            confidence = 0.0

        return {
            "is_injection": bool(findings),
            "confidence": round(confidence, 2),
            "categories": [item["category"] for item in findings],
            "findings": findings,
        }
