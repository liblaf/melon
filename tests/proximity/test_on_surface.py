from typing import Any

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Array, Float, Integer

from liblaf import melon


def on_surface_trimesh(mesh: Any, pcl: Any) -> tuple:
    mesh_tm: tm.Trimesh = melon.as_trimesh(mesh)
    pcl: pv.PointSet = melon.as_pointset(pcl)
    closest: Float[np.ndarray, "N 3"]
    distance: Float[np.ndarray, " N"]
    triangle_id: Integer[np.ndarray, " N"]
    closest, distance, triangle_id = mesh_tm.nearest.on_surface(pcl.points)
    return closest, distance, triangle_id


def test_on_surface() -> None:
    mesh: pv.PolyData = pv.examples.download_bunny()  # pyright: ignore[reportAssignmentType]
    mesh = melon.mesh_fix(mesh)
    rng: np.random.Generator = np.random.default_rng()
    points: Float[np.ndarray, "N 3"] = rng.uniform(
        low=mesh.bounds[::2], high=mesh.bounds[1::2], size=(1000, 3)
    )
    actual: melon.NearestPointOnSurfaceResult = melon.nearest_point_on_surface(
        mesh, points, distance_threshold=1.0, normal_threshold=None
    )
    desired_closest: Float[np.ndarray, "N 3"]
    desired_distance: Float[np.ndarray, " N"]
    desired_closest, desired_distance, _ = on_surface_trimesh(mesh, points)
    atol: float = 1e-2 * mesh.length
    np.testing.assert_allclose(actual.nearest, desired_closest, rtol=0.0, atol=atol)
    np.testing.assert_allclose(actual.distance, desired_distance, rtol=0.0, atol=atol)
    actual_nearest: Float[Array, "N 3"] = melon.barycentric_to_points(
        mesh.points[mesh.regular_faces[actual.triangle_id]], actual.barycentric
    )
    np.testing.assert_allclose(actual_nearest, desired_closest, rtol=0.0, atol=atol)
