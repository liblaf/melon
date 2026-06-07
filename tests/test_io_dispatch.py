from __future__ import annotations

from pathlib import Path
from typing import cast

import meshio
import numpy as np
import pytest
import pyvista as pv

from liblaf.melon import io
from liblaf.melon.io.abc import ConverterDispatcher, ReaderDispatcher, WriterDispatcher


def test_converter_dispatcher_keeps_target_instances_and_registers_sources() -> None:
    converter: ConverterDispatcher[str] = ConverterDispatcher(str)

    assert converter("ready") == "ready"

    @converter.register(int)
    def _from_int(obj: int, /, **kwargs: object) -> str:
        del kwargs
        return f"value={obj}"

    assert converter(3, ignored=True) == "value=3"
    with pytest.raises(NotImplementedError):
        converter(object())


def test_reader_dispatcher_prefers_suffix_registration_over_fallback(
    tmp_path: Path,
) -> None:
    reader: ReaderDispatcher[str] = ReaderDispatcher(str)

    @reader.register((".melon",))
    def _read_melon(path: Path, /, **kwargs: object) -> str:
        return f"known:{path.suffix}:{kwargs['flavor']}"

    @reader.register_fallback
    def _read_fallback(path: Path, /, **kwargs: object) -> str:
        del kwargs
        return f"fallback:{path.suffix}"

    assert reader(tmp_path / "sample.melon", flavor="sweet") == "known:.melon:sweet"
    assert reader(tmp_path / "sample.mesh") == "fallback:.mesh"


def test_writer_dispatcher_creates_parent_directories_and_dispatches_by_type(
    tmp_path: Path,
) -> None:
    writer: WriterDispatcher[str] = WriterDispatcher()

    @writer.register(str, (".txt",))
    def _write_text(obj: str, path: Path, /, **kwargs: object) -> None:
        del kwargs
        path.write_text(obj)

    path: Path = tmp_path / "nested" / "value.txt"

    writer("hello", path)

    assert path.read_text() == "hello"
    with pytest.raises(NotImplementedError):
        writer(cast("str", 1), tmp_path / "value.txt")
    with pytest.raises(KeyError):
        writer("hello", tmp_path / "value.unknown")


def test_meshio_triangle_converts_to_trimesh_and_polydata() -> None:
    mesh: meshio.Mesh = meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        cells=[("triangle", np.array([[0, 1, 2]]))],
    )

    trimesh = io.as_trimesh(mesh, process=False)
    polydata = io.as_polydata(trimesh)

    assert trimesh.vertices.shape == (3, 3)
    assert trimesh.faces.tolist() == [[0, 1, 2]]
    assert isinstance(polydata, pv.PolyData)
    assert polydata.n_cells == 1
