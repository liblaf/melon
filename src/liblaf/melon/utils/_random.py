import random
import string


def random_name(prefix: str = "", length: int = 8) -> str:
    suffix: str = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=length)  # noqa: S311
    )
    return f"{prefix}{suffix}"
