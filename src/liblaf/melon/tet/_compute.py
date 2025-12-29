from typing import Any

import einops
import jax
import numpy as np
import pyvista as pv

from liblaf.melon import io, utils


def compute_volume(mesh: Any) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = io.as_unstructured_grid(mesh)
    result: pv.UnstructuredGrid = mesh.compute_cell_sizes(
        length=False, area=False, volume=True
    )  # pyright: ignore[reportAssignmentType]
    return result


def compute_point_normals(
    mesh: Any,
    *,
    flip_normals: bool = False,
    consistent_normals: bool = True,
    auto_orient_normals: bool = False,
) -> pv.UnstructuredGrid:
    point_id_name: str = utils.random_name("_PointId_")
    mesh: pv.UnstructuredGrid = io.as_unstructured_grid(mesh)
    mesh.point_data[point_id_name] = np.arange(mesh.n_points)
    surface: pv.PolyData = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    surface.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=False,
        flip_normals=flip_normals,
        consistent_normals=consistent_normals,
        auto_orient_normals=auto_orient_normals,
        inplace=True,
    )
    mesh.point_data["Normals"] = np.zeros(
        (mesh.n_points, 3), surface.point_normals.dtype
    )
    mesh.point_data["Normals"][surface.point_data[point_id_name]] = (
        surface.point_normals
    )
    del mesh.point_data[point_id_name]
    return mesh


def compute_point_volume(mesh: Any) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = compute_volume(mesh)
    mesh.point_data["Volume"] = jax.ops.segment_sum(  # pyright: ignore[reportArgumentType]
        einops.repeat(mesh.cell_data["Volume"], "c -> (c p)", p=4),
        mesh.cells_dict[pv.CellType.TETRA].flatten(),  # pyright: ignore[reportArgumentType]
        num_segments=mesh.n_points,
    )
    return mesh
