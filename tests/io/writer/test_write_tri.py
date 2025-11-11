from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import trimesh as tm

from liblaf.melon import io


@pytest.mark.parametrize("suffix", [".obj", ".ply", ".vtp"])
def test_write_polydata(suffix: str, tmp_path: Path) -> None:
    mesh: pv.PolyData = pv.examples.download_bunny_coarse()  # pyright: ignore[reportAssignmentType]
    file: Path = tmp_path / f"mesh{suffix}"
    io.save(file, mesh)
    loaded: pv.PolyData = io.load_polydata(file)
    np.testing.assert_allclose(loaded.points, mesh.points)
    np.testing.assert_array_equal(loaded.regular_faces, mesh.regular_faces)


@pytest.mark.parametrize("suffix", [".obj", ".ply"])
def test_write_trimesh(suffix: str, tmp_path: Path) -> None:
    mesh_pv: pv.PolyData = pv.examples.download_bunny_coarse()  # pyright: ignore[reportAssignmentType]
    mesh: tm.Trimesh = io.as_trimesh(mesh_pv)
    file: Path = tmp_path / f"mesh{suffix}"
    io.save(file, mesh)
    loaded: tm.Trimesh = io.load_trimesh(file)
    np.testing.assert_allclose(loaded.vertices, mesh.vertices, atol=1e-7 * mesh.scale)
    np.testing.assert_array_equal(loaded.faces, mesh.faces)
