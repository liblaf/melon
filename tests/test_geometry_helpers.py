from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from liblaf.melon import io, tet, tri, xfer
from liblaf.melon.utils import filter_kwargs, pick, temporary_array


def _two_triangle_polydata() -> pv.PolyData:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([3, 0, 1, 2, 3, 0, 2, 3])
    mesh = pv.PolyData(points, faces)
    mesh.cell_data["GroupId"] = np.array([0, 1], dtype=np.int32)
    mesh.field_data["GroupName"] = np.array(["front", "back"])
    return mesh


def _tetra_grid(indices: list[int]) -> pv.UnstructuredGrid:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([4, *indices])
    cell_types = np.array([pv.CellType.TETRA])
    return pv.UnstructuredGrid(cells, cell_types, points)


def test_landmark_sidecar_uses_default_suffix_and_round_trips(
    tmp_path: Path,
) -> None:
    mesh_path: Path = tmp_path / "surface.obj"
    landmarks = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert io.load_landmarks(mesh_path).shape == (0, 3)

    io.save_landmarks(landmarks, mesh_path)

    assert (tmp_path / "surface.landmarks.json").exists()
    np.testing.assert_allclose(io.load_landmarks(mesh_path), landmarks)


def test_polygon_sidecar_accepts_boolean_masks_and_indices(tmp_path: Path) -> None:
    mask = np.array([False, True, False, True])
    path: Path = tmp_path / "polygons.json"

    io.save_polygons(mask, path)
    np.testing.assert_array_equal(io.load_polygons(path), np.array([1, 3], np.int32))

    io.save_polygons(np.array([0, 2], np.int32), path)
    np.testing.assert_array_equal(io.load_polygons(path), np.array([0, 2], np.int32))


def test_temporary_array_removes_data_after_success_and_failure() -> None:
    mesh = pv.PolyData(np.array([[0.0, 0.0, 0.0]]))

    with temporary_array(mesh.point_data, np.array([1.0]), name="tmp_") as name:
        assert name in mesh.point_data
    assert name not in mesh.point_data

    failed_name: dict[str, str] = {}

    def _raise_inside_temporary_array() -> None:
        with temporary_array(mesh.point_data, np.array([2.0]), name="tmp_") as name:
            failed_name["value"] = name
            message = "boom"
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="boom"):
        _raise_inside_temporary_array()
    assert failed_name["value"] not in mesh.point_data


def test_pick_and_filter_kwargs_keep_supported_inputs_only() -> None:
    def _target(alpha: int, *, beta: int) -> None:
        del alpha, beta

    assert pick({"alpha", "gamma"}, {"alpha": 1, "beta": 2}) == {"alpha": 1}
    assert filter_kwargs(_target, {"alpha": 1, "beta": 2, "gamma": 3}) == {
        "alpha": 1,
        "beta": 2,
    }


def test_group_selection_extraction_and_edge_lengths() -> None:
    mesh = _two_triangle_polydata()

    np.testing.assert_array_equal(tri.select_groups(mesh, "back"), [False, True])
    np.testing.assert_array_equal(
        tri.select_groups(mesh, 0, invert=True), [False, True]
    )

    extracted = tri.extract_groups(mesh, "back")
    lengths = np.sort(tri.edge_length(mesh))

    assert extracted.n_cells == 1
    np.testing.assert_allclose(lengths, [1.0, 1.0, 1.0, 1.0, np.sqrt(2.0)])


def test_tetra_winding_flip_and_repair() -> None:
    negative = _tetra_grid([0, 2, 1, 3])

    fixed = tet.fix_winding(negative)
    volumes = fixed.compute_cell_sizes(length=False, area=False, volume=True).cell_data[
        "Volume"
    ]

    assert volumes[0] > 0.0


def test_tri_point_to_tet_point_fills_unmatched_target_points() -> None:
    source = pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    source.point_data["value"] = np.array([10.0, 20.0])
    target = _tetra_grid([0, 1, 2, 3])

    result = xfer.tri_point_to_tet_point(source, target, {"value": -1.0})

    np.testing.assert_allclose(result.point_data["value"], [10.0, 20.0, -1.0, -1.0])
