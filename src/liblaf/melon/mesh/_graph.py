from typing import Any

import networkx as nx
import numpy as np
import polars as pl
import pyvista as pv
from jaxtyping import Integer

from liblaf.melon import io


def graph(mesh: Any) -> nx.Graph:
    mesh: pv.PolyData | pv.UnstructuredGrid = io.as_mesh(mesh).copy()
    mesh.point_data["__point_id"] = np.arange(mesh.n_points)
    edges: pv.PolyData = mesh.extract_all_edges()  # pyright: ignore[reportAssignmentType]
    edges = edges.compute_cell_sizes()  # pyright: ignore[reportAssignmentType]
    point_id: Integer[np.ndarray, " P"] = edges.point_data["__point_id"]
    lines: Integer[np.ndarray, "E 2"] = edges.lines.reshape(edges.n_lines, 3)[:, 1:]
    edgelist: pl.DataFrame = pl.from_dict(
        {
            "source": point_id[lines[:, 0]],
            "target": point_id[lines[:, 1]],
            "length": edges.cell_data["Length"],
        }  # pyright: ignore[reportArgumentType]
    )
    return nx.from_pandas_edgelist(edgelist, edge_attr="length")
