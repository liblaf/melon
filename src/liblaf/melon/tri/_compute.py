from typing import Any

import einops
import jax
import pyvista as pv

from liblaf.melon import io


def compute_area(mesh: Any) -> pv.PolyData:
    mesh: pv.PolyData = io.as_polydata(mesh)
    result: pv.PolyData = mesh.compute_cell_sizes(length=False, area=True, volume=False)  # pyright: ignore[reportAssignmentType]
    return result


def compute_point_area(mesh: Any) -> pv.PolyData:
    mesh: pv.PolyData = compute_area(mesh)
    mesh.point_data["Area"] = jax.ops.segment_sum(  # pyright: ignore[reportArgumentType]
        einops.repeat(mesh.cell_data["Area"], "c -> (c p)", p=3),
        mesh.regular_faces.flatten(),
        num_segments=mesh.n_points,
    )
    return mesh
