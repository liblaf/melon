from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Key


def sample_barycentric_coords(
    shape: Sequence[int], *, seed: int | ArrayLike = 0
) -> Float[Array, "*N D"]:
    n_samples: list[int]
    dim: int
    *n_samples, dim = shape
    key: Key = jax.random.key(seed)
    coords: Float[Array, "*N D-1"] = jax.random.uniform(key, (*n_samples, dim - 1))
    coords: Float[Array, "*N D-1"] = jnp.sort(coords, axis=-1)
    coords: Float[Array, "*N D"] = jnp.diff(coords, axis=-1, prepend=0, append=1)
    return coords
