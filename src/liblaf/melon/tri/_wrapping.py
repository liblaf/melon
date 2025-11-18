import logging
from typing import Any

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.melon import io

logger: logging.Logger = logging.getLogger(__name__)


def fast_wrapping(
    source: Any,
    target: Any,
    *,
    source_landmarks: Float[ArrayLike, "L 3"] | None = None,
    target_landmarks: Float[ArrayLike, "L 3"] | None = None,
    free_polygons_floating: Integer[ArrayLike, " F"] | None = None,
    verbose: bool = True,
) -> pv.PolyData:
    from liblaf.melon.ext import wrap

    if source_landmarks is not None and target_landmarks is not None:
        matrix: Float[np.ndarray, "4 4"]
        transformed: Float[np.ndarray, "L 3"]
        cost: float
        matrix, transformed, cost = tm.registration.procrustes(
            source_landmarks, target_landmarks
        )
        logger.debug("procrustes cost: %g", cost)
        source: pv.PolyData = io.as_polydata(source)
        source = source.transform(matrix)  # pyright: ignore[reportAssignmentType]
        source_landmarks = transformed
    result: pv.PolyData = wrap.fast_wrapping(
        source,
        target,
        source_landmarks=source_landmarks,
        target_landmarks=target_landmarks,
        free_polygons_floating=free_polygons_floating,
        verbose=verbose,
    )
    return result
