
import re


def clean_text(text: str) -> str:
    """Normalize whitespace in extracted document text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Normalize Windows and old-style line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove spaces and tabs at the beginning/end of lines.
    text = "\n".join(
        line.strip()
        for line in text.split("\n")
    )

    # Reduce excessive blank lines while preserving paragraphs.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize repeated horizontal whitespace.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
