import numpy as np
import pyvista as pv
from jaxtyping import Float


def edge_length(mesh: pv.PolyData) -> Float[np.ndarray, " E"]:
    edges: pv.PolyData = mesh.extract_all_edges()
    edges: pv.PolyData = edges.compute_cell_sizes(length=True)
    return edges.cell_data["Length"]
