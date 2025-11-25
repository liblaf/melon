import numpy as np
import pyvista as pv
from jaxtyping import Array, Integer

from liblaf import melon


def cell_neighbors_pyvista(mesh: pv.UnstructuredGrid) -> Integer[np.ndarray, "C 2"]:
    neighbors_list: list[tuple[int, int]] = []
    for ind in range(mesh.n_cells):
        neighbors_list.extend(
            (ind, neighbor) for neighbor in mesh.cell_neighbors(ind, "faces")
        )
    neighbors: Integer[np.ndarray, "N 2"] = np.array(neighbors_list, dtype=np.int32)
    neighbors.sort(axis=-1)
    neighbors = np.unique(neighbors, axis=0)
    return neighbors


def test_cell_neighbors() -> None:
    mesh: pv.UnstructuredGrid = pv.examples.download_letter_a()  # pyright: ignore[reportAssignmentType]
    actual: Integer[Array, "N 2"] = melon.cell_neighbors(mesh)
    expected: Integer[np.ndarray, "N 2"] = cell_neighbors_pyvista(mesh)
    np.testing.assert_array_equal(actual, expected)
