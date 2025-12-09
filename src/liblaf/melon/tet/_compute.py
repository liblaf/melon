from typing import Any

import einops
import jax
import pyvista as pv

from liblaf.melon import io


def compute_volume(mesh: Any) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = io.as_unstructured_grid(mesh)
    result: pv.UnstructuredGrid = mesh.compute_cell_sizes(
        length=False, area=False, volume=True
    )  # pyright: ignore[reportAssignmentType]
    return result


def compute_point_volume(mesh: Any) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = compute_volume(mesh)
    mesh.point_data["Volume"] = jax.ops.segment_sum(  # pyright: ignore[reportArgumentType]
        einops.repeat(mesh.cell_data["Volume"], "c -> (c p)", p=4),
        mesh.cells_dict[pv.CellType.TETRA].flatten(),  # pyright: ignore[reportArgumentType]
        num_segments=mesh.n_points,
    )
    return mesh
