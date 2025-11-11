import numpy as np
import pyvista as pv

from liblaf.melon.ext import mesh_fix


def test_mesh_fix() -> None:
    mesh: pv.PolyData = pv.examples.download_bunny_coarse()  # pyright: ignore[reportAssignmentType]
    mesh.field_data["foo"] = np.arange(3)
    fixed: pv.PolyData = mesh_fix(mesh, check=True)
    np.testing.assert_array_equal(mesh.field_data["foo"], fixed.field_data["foo"])
