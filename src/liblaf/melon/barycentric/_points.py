import einops
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


def barycentric_to_points(
    cells: Float[ArrayLike, "*N B D"], barycentric: Float[ArrayLike, "*N B"]
) -> Float[Array, "*N D"]:
    cells: Float[Array, "*N B D"] = jnp.asarray(cells)
    barycentric: Float[Array, "*N B"] = jnp.asarray(barycentric)
    points: Float[Array, "*N D"] = einops.einsum(
        barycentric, cells, "... B, ... B D -> ... D"
    )
    return points
