from collections.abc import Sequence
from typing import Any, Literal

import trimesh as tm


def intersection(
    meshes: Sequence[Any],
    *,
    engine: Literal["blender", "manifold"] | None = None,
    check_volume: bool = True,
    **kwargs,
) -> tm.Trimesh:
    return tm.boolean.intersection(
        meshes, engine=engine, check_volume=check_volume, **kwargs
    )
