import numpy as np
import pyvista as pv
from jaxtyping import Float


def edge_length(mesh: pv.PolyData) -> Float[np.ndarray, " E"]:
    """Compute lengths for all extracted edges of a triangular surface.

    Args:
        mesh: Surface mesh.

    Returns:
        Edge lengths reported by PyVista's cell-size filter.
    """
    edges: pv.PolyData = mesh.extract_all_edges()
    edges: pv.PolyData = edges.compute_cell_sizes(length=True)
    return edges.cell_data["Length"]
