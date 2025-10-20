from typing import Any

import numpy as np
import pyvista as pv
from jaxtyping import Float

from liblaf.melon import io


def compute_edge_lengths(mesh: Any) -> Float[np.ndarray, " E"]:
    mesh: pv.PolyData = io.as_polydata(mesh)
    edges: pv.PolyData = mesh.extract_all_edges()  # pyright: ignore[reportAssignmentType]
    edges = edges.compute_cell_sizes()  # pyright: ignore[reportAssignmentType]
    return edges.cell_data["Length"]
