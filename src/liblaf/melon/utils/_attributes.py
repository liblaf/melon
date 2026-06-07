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
    """Attach a temporary PyVista data array and remove it on exit.

    Args:
        attributes: Point, cell, or field data container to mutate.
        data: Optional array value to store under the generated name.
        name: Prefix for the generated array name.
        length: Number of random suffix characters to append.

    Yields:
        The generated array name.
    """
    suffix: str = "".join(random.choices(_ALPHABET, k=length))  # noqa: S311
    name: str = f"{name}{suffix}"
    if data is not None:
        attributes[name] = data
    try:
        yield name
    finally:
        attributes.pop(name, None)
