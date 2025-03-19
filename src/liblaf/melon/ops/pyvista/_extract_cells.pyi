from typing import overload

import pyvista as pv
from jaxtyping import Bool, Integer
from numpy.typing import ArrayLike

@overload
def extract_cells(
    mesh: pv.PolyData, selection: Bool[ArrayLike, " C"] | Integer[ArrayLike, " N"]
) -> pv.PolyData: ...
@overload
def extract_cells(
    mesh: pv.UnstructuredGrid,
    selection: Bool[ArrayLike, " C"] | Integer[ArrayLike, " N"],
) -> pv.UnstructuredGrid: ...
