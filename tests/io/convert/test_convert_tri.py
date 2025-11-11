import numpy as np
import pyvista as pv
import trimesh as tm

from liblaf.melon import io


def test_polydata_to_trimesh() -> None:
    polydata: pv.PolyData = pv.examples.download_bunny_coarse()  # pyright: ignore[reportAssignmentType]
    trimesh: tm.Trimesh = io.as_trimesh(polydata)
    assert isinstance(trimesh, tm.Trimesh)
    np.testing.assert_allclose(trimesh.vertices, polydata.points)
    np.testing.assert_array_equal(trimesh.faces, polydata.regular_faces)


def test_trimesh_to_polydata() -> None:
    polydata: pv.PolyData = pv.examples.download_bunny_coarse()  # pyright: ignore[reportAssignmentType]
    trimesh: tm.Trimesh = io.as_trimesh(polydata)
    polydata: pv.PolyData = io.as_polydata(trimesh)
    assert isinstance(polydata, pv.PolyData)
    np.testing.assert_allclose(polydata.points, trimesh.vertices)
    np.testing.assert_array_equal(polydata.regular_faces, trimesh.faces)
