import einops
import numpy as np
from jaxtyping import Float
from scipy.stats import qmc


def sample(n: int, d: int, *, scramble: bool = False) -> Float[np.ndarray, "n d"]:
    sampler: qmc.QMCEngine = qmc.Sobol(d=d - 1, scramble=scramble)
    sample: Float[np.ndarray, "n d-1"] = sampler.random(n)
    sample.sort(axis=-1)
    sample: Float[np.ndarray, "n d"] = np.diff(sample, axis=-1, prepend=0.0, append=1.0)
    return sample


def to_points[T](
    data: Float[T, "c b *d"], barycentric: Float[T, "c b"]
) -> Float[T, " c *d"]:
    return einops.einsum(data, barycentric, "c b ..., c b -> c ...")
