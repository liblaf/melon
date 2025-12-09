import numpy as np
import pyvista as pv
from jaxtyping import Float


def compute_edges_length(obj: pv.DataSet) -> Float[np.ndarray, " edges"]:
    edges: pv.PolyData = obj.extract_all_edges()  # pyright: ignore[reportAssignmentType]
    edges = edges.compute_cell_sizes(length=True, area=False, volume=False)  # pyright: ignore[reportAssignmentType]
    return edges.cell_data["Length"]
