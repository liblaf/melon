from typing import Any

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Bool, Float

from liblaf import melon


def contains_trimesh(mesh: Any, pcl: Any) -> Bool[np.ndarray, " N"]:
    mesh_tm: tm.Trimesh = melon.as_trimesh(mesh)
    pcl: pv.PointSet = melon.as_pointset(pcl)
    return mesh_tm.contains(pcl.points)


def test_contains() -> None:
    mesh: pv.PolyData = pv.examples.download_bunny()  # pyright: ignore[reportAssignmentType]
    mesh = melon.mesh_fix(mesh)
    rng: np.random.Generator = np.random.default_rng()
    points: Float[np.ndarray, "N 3"] = rng.uniform(
        low=mesh.bounds[::2], high=mesh.bounds[1::2], size=(1000, 3)
    )
    actual: Bool[np.ndarray, " N"] = melon.tri.contains(mesh, points)
    desired: Bool[np.ndarray, " N"] = contains_trimesh(mesh, points)
    n_diff: int = np.count_nonzero(actual != desired)  # pyright: ignore[reportAssignmentType]
    assert n_diff < 1e-2 * points.shape[0]
