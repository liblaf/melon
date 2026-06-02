import contextlib
import random
import string
from collections.abc import Generator

import pyvista as pv
from jaxtyping import Float
from numpy.typing import ArrayLike

_ALPHABET: str = string.ascii_lowercase + string.digits + "_"


@contextlib.contextmanager
def temporary_array(
    attributes: pv.DataSetAttributes,
    data: Float[ArrayLike, "..."] | None = None,
    name: str = "",
    length: int = 8,
) -> Generator[str]:
    suffix: str = "".join(random.choices(_ALPHABET, k=length))  # noqa: S311
    name: str = f"{name}{suffix}"
    if data is not None:
        attributes[name] = data
    yield name
    attributes.pop(name, None)
