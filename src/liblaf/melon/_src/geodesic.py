from typing import Any

import numpy as np
import potpourri3d as pp3d
import pyvista as pv
from jaxtyping import ArrayLike, Float, Integer

from liblaf.melon import io


def geodesic_distance(
    mesh: Any, v_ind: Integer[ArrayLike, "*v"]
) -> Float[np.ndarray, " p *v"]:
    mesh: pv.PolyData = io.as_polydata(mesh).triangulate(inplace=False)  # pyright: ignore[reportAssignmentType]
    v_ind: Integer[np.ndarray, "*v"] = np.asarray(v_ind)
    v_ind_flat: Integer[np.ndarray, " V"] = np.ravel(v_ind)
    solver = pp3d.MeshHeatMethodDistanceSolver(mesh.points, mesh.regular_faces)
    dist_flat: Float[np.ndarray, "p v"] = np.stack(
        [solver.compute_distance(vi) for vi in v_ind_flat], axis=-1
    )
    return dist_flat.reshape((mesh.n_points, *v_ind.shape))


def geodesic_path(mesh: Any, v_start: int, v_end: int) -> pv.PolyData:
    mesh: pv.PolyData = io.as_polydata(mesh).triangulate(inplace=False)  # pyright: ignore[reportAssignmentType]
    solver = pp3d.EdgeFlipGeodesicSolver(mesh.points, mesh.regular_faces)
    points: Float[np.ndarray, "p 3"] = solver.find_geodesic_path(v_start, v_end)
    return pv.lines_from_points(points)
