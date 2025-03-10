from collections.abc import Mapping
from typing import Any

import numpy as np
from jaxtyping import Bool, Float, Integer

from liblaf import melon


def transfer_components(
    target: Any,
    components: Mapping[str, Any],
    *,
    proximity: melon.NearestAlgorithm | None = None,
) -> dict[str, Bool[np.ndarray, " target.n_points"]]:
    nearest: dict[str, melon.NearestResult] = {}
    for name, component in components.items():
        nearest[name] = melon.nearest(component, target, algo=proximity)
    distance: Float[np.ndarray, "n_components target.n_points"] = np.column_stack(
        [r.distance for r in nearest.values()]
    )
    component: Integer[np.ndarray, " target.n_points"] = np.argmin(distance, axis=1)
    point_data: dict[str, Bool[np.ndarray, " target.n_points"]] = {}
    for component_id, component_name in enumerate(components):
        point_data[f"is-{component_name}"] = component == component_id
    return point_data
