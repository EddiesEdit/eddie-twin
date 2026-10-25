
class InputGuard:
    """Validate user input before processing."""

    def __init__(self, max_length: int = 8_000) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        self.max_length = max_length

    def validate(self, user_input: str) -> str:
        """Return normalized input or raise ValueError."""

        if not isinstance(user_input, str):
            raise ValueError("Input must be a string.")

        clean_input = user_input.strip()

        if not clean_input:
            raise ValueError("Input cannot be empty.")

        if len(clean_input) > self.max_length:
            raise ValueError(
                f"Input exceeds the maximum length of "
                f"{self.max_length} characters."
            )

        if "\x00" in clean_input:
            raise ValueError("Input contains an invalid null character.")

        return clean_input
