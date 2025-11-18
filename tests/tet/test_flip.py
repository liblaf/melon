import hypothesis
import hypothesis.extra.numpy as hnp
import numpy as np
import pyvista as pv
from hypothesis import strategies as st

from liblaf import melon


@hypothesis.given(data=st.data())
def test_tet_fix_winding(data: st.DataObject) -> None:
    mesh: pv.UnstructuredGrid = pv.examples.download_letter_a()  # pyright: ignore[reportAssignmentType]
    mesh = mesh.compute_cell_sizes()  # pyright: ignore[reportAssignmentType]
    assert np.all(mesh.cell_data["Volume"] >= 0)
    mask: np.ndarray = data.draw(hnp.arrays(np.bool, (mesh.n_cells,)))
    mesh: pv.UnstructuredGrid = melon.tet.flip(mesh, mask)
    mesh = mesh.compute_cell_sizes()  # pyright: ignore[reportAssignmentType]
    assert np.all(mesh.cell_data["Volume"][~mask] >= 0)
    assert np.all(mesh.cell_data["Volume"][mask] <= 0)
    mesh = melon.tet.fix_winding(mesh)
    mesh = mesh.compute_cell_sizes()  # pyright: ignore[reportAssignmentType]
    assert np.all(mesh.cell_data["Volume"] >= 0)
