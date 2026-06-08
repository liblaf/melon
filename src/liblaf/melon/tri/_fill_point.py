from collections.abc import Iterable

import numpy as np
import pyvista as pv
import scipy.sparse
import trimesh as tm
from jaxtyping import Bool, Float, Integer


def fill_point(
    mesh: pv.PolyData,
    mask: Bool[np.ndarray, " P"],
    names: Iterable[str] | None = None,
    *,
    limit: float = 0.01,
) -> pv.PolyData:
    if names is None:
        names: list[str] = mesh.point_data.keys()
    csgraph: scipy.sparse.coo_array = _make_csgraph(mesh)
    _dist_matrix, _predecessors, sources = scipy.sparse.csgraph.dijkstra(
        csgraph,
        directed=False,
        indices=np.flatnonzero(~mask),
        return_predecessors=True,
        limit=limit * mesh.length,
        min_only=True,
    )
    mask &= sources >= 0
    for name in names:
        data: Float[np.ndarray, "S ..."] = mesh.point_data[name]
        data[mask] = data[sources[mask]]
        mesh.point_data[name] = data
    return mesh


def _make_csgraph(mesh: pv.PolyData) -> scipy.sparse.coo_array:
    n_points: int = mesh.n_points
    mesh: tm.Trimesh = pv.to_trimesh(mesh, triangulate=True)
    edges: Integer[np.ndarray, "E 2"] = mesh.edges_unique
    lengths: Float[np.ndarray, " E"] = mesh.edges_unique_length
    row: Integer[np.ndarray, "2E"] = np.concat([edges[:, 0], edges[:, 1]])
    col: Integer[np.ndarray, "2E"] = np.concat([edges[:, 1], edges[:, 0]])
    data: Float[np.ndarray, "2E"] = np.concat([lengths, lengths])
    return scipy.sparse.coo_array((data, (row, col)), (n_points, n_points))
