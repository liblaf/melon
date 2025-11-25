import itertools
from collections.abc import Iterable, Mapping

import einops
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from jaxtyping import Array, Integer
from liblaf.grapes.bench import Bencher, BenchResults

from liblaf import grapes


def main() -> None:
    grapes.logging.init()
    bencher = Bencher()

    @bencher.setup
    def _setup() -> Iterable[tuple[tuple[pv.UnstructuredGrid], Mapping]]:
        mesh: pv.UnstructuredGrid = pv.examples.download_letter_a()  # pyright: ignore[reportAssignmentType]
        yield (mesh,), {}

    @bencher.size
    def _(mesh: pv.UnstructuredGrid) -> int:
        return mesh.n_cells

    @bencher.bench(label="PyVista")
    def _(mesh: pv.UnstructuredGrid) -> Integer[np.ndarray, "C 2"]:
        neighbors_list: list[tuple[int, int]] = []
        for ind in range(mesh.n_cells):
            neighbors_list.extend(
                (ind, neighbor) for neighbor in mesh.cell_neighbors(ind, "faces")
            )
        neighbors: Integer[np.ndarray, "N 2"] = np.array(neighbors_list, dtype=np.int32)
        neighbors.sort(axis=-1)
        neighbors = np.unique(neighbors, axis=0)
        return neighbors

    @bencher.bench(label="Custom")
    def _cell_neighbors_custom(mesh: pv.UnstructuredGrid) -> Integer[Array, "N 2"]:
        cells: Integer[Array, "T 4"] = jnp.asarray(
            mesh.cells_dict[pv.CellType.TETRA]  # pyright: ignore[reportArgumentType]
        )
        combinations: Integer[Array, "4 3"] = jnp.asarray(
            list(itertools.combinations(range(4), 3))
        )
        faces: Integer[Array, "T 4 3"] = cells[:, combinations]
        faces = jnp.sort(faces, axis=-1)
        cell_idx: Integer[Array, " T*4"] = einops.repeat(
            jnp.arange(mesh.n_cells), "T -> (T 4)", T=mesh.n_cells
        )
        faces: Integer[Array, "T*4 3"] = einops.rearrange(faces, "T F V -> (T F) V")
        order: Integer[Array, " T*4"] = jnp.lexsort(faces.T)
        faces_sorted: Integer[Array, " T*4 3"] = faces[order]
        cell_idx_sorted: Integer[Array, " T*4"] = cell_idx[order]
        mask: Array = jnp.all(faces_sorted[:-1] == faces_sorted[1:], axis=-1)
        neighbors: Integer[Array, "N 2"] = jnp.stack(
            [cell_idx_sorted[:-1][mask], cell_idx_sorted[1:][mask]], axis=-1
        )
        neighbors = jnp.sort(neighbors, axis=-1)
        neighbors = jnp.unique(neighbors, axis=0)
        return neighbors

    results: BenchResults = bencher.run()
    results.plot()
    plt.show()

    for label, outputs in results.outputs.items():
        if label == "PyVista":
            continue
        for actual, expected in zip(outputs, results.outputs["PyVista"], strict=True):
            if actual is None or expected is None:
                continue
            actual: Integer[np.ndarray, "C 2"]
            expected: Integer[Array, "C 2"]
            ic(actual, expected)
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    main()
