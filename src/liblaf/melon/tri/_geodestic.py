import numpy as np
import potpourri3d as pp3d
import pyvista as pv
from jaxtyping import Float


def geodesic_path(mesh: pv.PolyData, v_start: int, v_end: int) -> pv.PolyData:
    solver = pp3d.EdgeFlipGeodesicSolver(mesh.points, mesh.regular_faces)
    points: Float[np.ndarray, "p 3"] = solver.find_geodesic_path(v_start, v_end)
    return pv.lines_from_points(points)
