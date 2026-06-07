from __future__ import annotations

import contextlib

import numpy as np
import pytest
import pyvista as pv
import torch

from liblaf.melon.tet import _volume_fraction as vf


def _single_tetra_grid() -> pv.UnstructuredGrid:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([4, 0, 1, 2, 3])
    cell_types = np.array([pv.CellType.TETRA])
    return pv.UnstructuredGrid(cells, cell_types, points)


def _triangle_surface() -> pv.PolyData:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2]])
    return pv.make_tri_mesh(points, faces)


def test_volume_fraction_handles_one_boundary_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def _contains_points(_mesh: object, points: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor([True, False, False, False], device=points.device)
        sample_ids = torch.arange(points.shape[0], device=points.device)
        return sample_ids % 2 == 0

    monkeypatch.setattr(vf, "_torch_device", contextlib.nullcontext)
    monkeypatch.setattr(vf, "_contains_points", _contains_points)

    fraction = vf.volume_fraction(
        _single_tetra_grid(), _triangle_surface(), n_samples=3, split_size=1
    )

    np.testing.assert_allclose(fraction, [0.5])
