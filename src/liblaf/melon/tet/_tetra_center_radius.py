import attrs
import numpy as np
import pyvista as pv
import torch
from jaxtyping import Float, Integer
from torch import Tensor


@attrs.frozen
class TetraCenterRadiusResult:
    center: Float[Tensor, "c 3"]
    radius: Float[Tensor, " c"]


def tetra_center_radius(mesh: pv.UnstructuredGrid) -> TetraCenterRadiusResult:
    points: Float[Tensor, "p 3"] = torch.tensor(mesh.points, dtype=torch.float32)
    centers: pv.PolyData = mesh.cell_centers()
    centers: Float[Tensor, "c 3"] = torch.tensor(centers.points, dtype=torch.float32)
    cells: Integer[np.ndarray, "c 4"] = mesh.cells_dict[pv.CellType.TETRA]  # ty:ignore[invalid-argument-type]
    cells: Integer[Tensor, "c 4"] = torch.tensor(cells, dtype=torch.int32)
    radius: Float[Tensor, " c"] = torch.amax(
        (centers[:, torch.newaxis, :] - points[cells]).norm(dim=-1), dim=-1
    )
    return TetraCenterRadiusResult(center=centers, radius=radius)
