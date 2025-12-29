from typing import Any

import numpy as np
import trimesh as tm
from jaxtyping import ArrayLike, Float

from liblaf.melon import io


def icp(
    source: Any,
    target: Any,
    *,
    n_samples: int = 10000,
    initial: Float[ArrayLike, "4 4"] | None = None,
    threshold: float = 1e-6,
    max_iterations: int = 100,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
) -> tuple[Float[np.ndarray, "4 4"], float]:
    source: tm.Trimesh = io.as_trimesh(source)
    target: tm.Trimesh = io.as_trimesh(target)
    matrix: Float[np.ndarray, "4 4"]
    cost: float
    matrix, _, cost = tm.registration.icp(
        source.sample(n_samples),
        target.sample(n_samples),
        initial=initial,
        threshold=threshold,
        max_iterations=max_iterations,
        reflection=reflection,
        translation=translation,
        scale=scale,
    )
    return matrix, cost
