from collections.abc import Container
from pathlib import Path

import pyvista as pv
import rich.pretty

from liblaf.melon import io

POLY_DATA_SUFFIXES: Container[str] = {".obj", ".off", ".ply", ".stl", ".vtp"}
UNSTRUCTURED_GRID_SUFFIXES: Container[str] = {".msh", ".vtu"}


def info(
    path: Path,
    /,
    *,
    point_data: bool = True,
    cell_data: bool = True,
    field_data: bool = True,
    field_data_values: bool = False,
    user_dict: bool = True,
) -> None:
    mesh: pv.PolyData | pv.UnstructuredGrid
    if path.suffix in POLY_DATA_SUFFIXES:
        mesh = io.load_polydata(path)
    elif path.suffix in UNSTRUCTURED_GRID_SUFFIXES:
        mesh = io.load_unstructured_grid(path)
    else:
        msg: str = f"Unsupported file format: {path.suffix}"
        raise ValueError(msg)
    rich.pretty.pprint(mesh)
    if point_data:
        rich.pretty.pprint(mesh.point_data)
    if cell_data:
        rich.pretty.pprint(mesh.cell_data)
    if field_data:
        rich.pretty.pprint(mesh.field_data)
    if field_data_values:
        for name, data in mesh.field_data.items():
            rich.print(f"{name}: ", end="")
            rich.pretty.pprint(data)
    if user_dict:
        rich.print("User Dict: ", end="")
        rich.pretty.pprint(mesh.user_dict)
