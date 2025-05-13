import einops
import numpy as np
from jaxtyping import Float
from numpy.typing import ArrayLike


def barycentric_to_points(
    cells: Float[ArrayLike, "*N B D"], barycentric: Float[ArrayLike, "*N B"]
) -> Float[np.ndarray, "*N D"]:
    cells: Float[np.ndarray, "*N B D"] = np.asarray(cells)
    barycentric: Float[np.ndarray, "*N B"] = np.asarray(barycentric)
    points: Float[np.ndarray, "*N D"] = einops.einsum(
        barycentric, cells, "... B, ... B D -> ... D"
    )
    return points
