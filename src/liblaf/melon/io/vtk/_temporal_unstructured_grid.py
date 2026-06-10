import functools
from pathlib import Path
from types import TracebackType
from typing import Self

import attrs
import h5py as h5
import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike, DTypeLike

from liblaf.melon.io.utils import h5repack


class FileWrapper:
    file: h5.File

    def require_group(self, name: str) -> h5.Group:
        return self.file.require_group(name)

    def require_dataset(
        self, name: str, shape: tuple[int, ...], dtype: DTypeLike | None = None
    ) -> h5.Dataset:
        shape: tuple[int, ...] = (0, *shape[1:])
        maxshape: tuple[int | None, ...] = (None, *shape[1:])
        dataset: h5.Dataset = self.file.require_dataset(
            name,
            shape,
            dtype,
            maxshape=maxshape,
            compression="gzip",
            compression_opts=4,
        )
        return dataset


@attrs.define
class StepsGroup(FileWrapper):
    file: h5.File

    @property
    def n_steps(self) -> int:
        group: h5.Group = self.file.require_group("/VTKHDF/Steps")
        return group.attrs["NSteps"]

    @n_steps.setter
    def n_steps(self, value: int) -> None:
        group: h5.Group = self.file.require_group("/VTKHDF/Steps")
        group.attrs["NSteps"] = value

    @functools.cached_property
    def values(self) -> h5.Dataset:
        return self.require_dataset("/VTKHDF/Steps/Values", (0,), np.float64)

    @functools.cached_property
    def part_offsets(self) -> h5.Dataset:
        return self.require_dataset("/VTKHDF/Steps/PartOffsets", (0,), np.int64)

    @functools.cached_property
    def number_of_parts(self) -> h5.Dataset:
        return self.require_dataset("/VTKHDF/Steps/NumberOfParts", (0,), np.int64)

    @functools.cached_property
    def point_offsets(self) -> h5.Dataset:
        return self.require_dataset("/VTKHDF/Steps/PointOffsets", (0,), np.int64)

    @functools.cached_property
    def cell_offsets(self) -> h5.Dataset:
        return self.require_dataset("/VTKHDF/Steps/CellOffsets", (0, 1), np.int64)

    @functools.cached_property
    def connectivity_id_offsets(self) -> h5.Dataset:
        return self.require_dataset(
            "/VTKHDF/Steps/ConnectivityIdOffsets", (0, 1), np.int64
        )

    def point_data_offsets(self, name: str) -> h5.Dataset:
        return self.require_dataset(
            f"/VTKHDF/Steps/PointDataOffsets/{name}", (0,), np.int64
        )

    def cell_data_offsets(self, name: str) -> h5.Dataset:
        return self.require_dataset(
            f"/VTKHDF/Steps/CellDataOffsets/{name}", (0,), np.int64
        )

    def field_data_offsets(self, name: str) -> h5.Dataset:
        return self.require_dataset(
            f"/VTKHDF/Steps/FieldDataOffsets/{name}", (0,), np.int64
        )

    def field_data_sizes(self, name: str) -> h5.Dataset:
        return self.require_dataset(
            f"/VTKHDF/Steps/FieldDataSizes/{name}", (0, 2), np.int64
        )


@attrs.define
class VTKHDFTemporalUnstructuredGridWriter(FileWrapper):
    _file: Path = attrs.field(converter=Path, alias="file")

    def __enter__(self) -> Self:
        self.file.__enter__()
        group: h5.Group = self.require_group("/VTKHDF")
        group.attrs["Version"] = (2, 5)
        group.attrs["Type"] = "UnstructuredGrid"
        self.n_steps = 0
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.file.__exit__(exc_type, exc_value, traceback)
        if exc_type is None:
            h5repack(self._file)

    @functools.cached_property
    def file(self) -> h5.File:
        return h5.File(self._file, "w")

    @functools.cached_property
    def number_of_connectivity_ids(self) -> h5.Dataset:
        return self.require_dataset("/VTKHDF/NumberOfConnectivityIds", (0,), np.int64)

    @functools.cached_property
    def number_of_points(self) -> h5.Dataset:
        return self.require_dataset("/VTKHDF/NumberOfPoints", (0,), np.int64)

    @functools.cached_property
    def number_of_cells(self) -> h5.Dataset:
        return self.require_dataset("/VTKHDF/NumberOfCells", (0,), np.int64)

    @functools.cached_property
    def points(self) -> h5.Dataset:
        return self.require_dataset("/VTKHDF/Points", (0, 3), np.float64)

    @functools.cached_property
    def connectivity(self) -> h5.Dataset:
        return self.require_dataset("/VTKHDF/Connectivity", (0,), np.int64)

    @functools.cached_property
    def offsets(self) -> h5.Dataset:
        return self.require_dataset("/VTKHDF/Offsets", (0,), np.int64)

    @functools.cached_property
    def types(self) -> h5.Dataset:
        return self.require_dataset("/VTKHDF/Types", (0,), np.uint8)

    def point_data(
        self, name: str, shape: tuple[int, ...], dtype: DTypeLike | None = None
    ) -> h5.Dataset:
        return self.require_dataset(f"/VTKHDF/PointData/{name}", shape, dtype)

    def cell_data(
        self, name: str, shape: tuple[int, ...], dtype: DTypeLike | None = None
    ) -> h5.Dataset:
        return self.require_dataset(f"/VTKHDF/CellData/{name}", shape, dtype)

    def field_data(
        self, name: str, shape: tuple[int, ...], dtype: DTypeLike | None = None
    ) -> h5.Dataset:
        return self.require_dataset(f"/VTKHDF/FieldData/{name}", shape, dtype)

    @functools.cached_property
    def steps(self) -> StepsGroup:
        return StepsGroup(self.file)

    @property
    def n_steps(self) -> int:
        return self.steps.n_steps

    @n_steps.setter
    def n_steps(self, value: int) -> None:
        self.steps.n_steps = value

    def _check_mesh(self, mesh: pv.UnstructuredGrid) -> None:
        if self.steps.n_steps == 0:
            for attributes in [mesh.point_data, mesh.cell_data, mesh.field_data]:
                for name in attributes:
                    for invalid_char in ["/", "."]:
                        if invalid_char in name:
                            msg: str = f"Invalid attribute name: {name!r}"
                            raise ValueError(msg)
        else:
            _check_attributes(self.require_group("/VTKHDF/PointData"), mesh.point_data)
            _check_attributes(self.require_group("/VTKHDF/CellData"), mesh.cell_data)
            _check_attributes(self.require_group("/VTKHDF/FieldData"), mesh.field_data)

    def _append_points(self, mesh: pv.UnstructuredGrid) -> None:
        _append_offsets(self.points, self.steps.point_offsets, mesh.points)

    def _append_cells(self, mesh: pv.UnstructuredGrid) -> None:
        if (
            self.steps.n_steps > 0
            and self.number_of_points[-1] == mesh.n_points
            and self.number_of_cells[-1] == mesh.n_cells
            and self.number_of_connectivity_ids[-1] == mesh.cell_connectivity.size
            and _tail_equal(self.types, mesh.celltypes)
            and _tail_equal(self.offsets, mesh.offset)
            and _tail_equal(self.connectivity, mesh.cell_connectivity)
        ):
            _append(self.steps.cell_offsets, self.steps.cell_offsets[-1])
            _append(
                self.steps.connectivity_id_offsets,
                self.steps.connectivity_id_offsets[-1],
            )
            _append(self.steps.part_offsets, self.steps.part_offsets[-1])
            _append(self.steps.number_of_parts, self.steps.number_of_parts[-1])
        else:
            cell_offset: int = self.types.len()
            connectivity_id_offset: int = self.connectivity.len()
            part_offset: int = self.number_of_points.len()
            _append(self.types, mesh.celltypes)
            _append(self.offsets, mesh.offset)
            _append(self.connectivity, mesh.cell_connectivity)
            _append(self.number_of_points, mesh.n_points)
            _append(self.number_of_cells, mesh.n_cells)
            _append(self.number_of_connectivity_ids, mesh.cell_connectivity.size)
            _append(self.steps.cell_offsets, cell_offset)
            _append(self.steps.connectivity_id_offsets, connectivity_id_offset)
            _append(self.steps.part_offsets, part_offset)
            _append(self.steps.number_of_parts, 1)

    def _append_point_data(self, mesh: pv.UnstructuredGrid) -> None:
        for name, arr_pv in mesh.point_data.items():
            arr: np.ndarray = _sanitize_array(arr_pv)
            dataset: h5.Dataset = self.point_data(name, arr.shape, arr.dtype)
            offsets: h5.Dataset = self.steps.point_data_offsets(name)
            _append_offsets(dataset, offsets, arr)

    def _append_cell_data(self, mesh: pv.UnstructuredGrid) -> None:
        for name, arr_pv in mesh.cell_data.items():
            arr: np.ndarray = _sanitize_array(arr_pv)
            dataset: h5.Dataset = self.cell_data(name, arr.shape, arr.dtype)
            offsets: h5.Dataset = self.steps.cell_data_offsets(name)
            _append_offsets(dataset, offsets, arr)

    def _append_field_data(self, mesh: pv.UnstructuredGrid) -> None:
        for name, arr_pv in mesh.field_data.items():
            arr: np.ndarray = _sanitize_array(arr_pv)
            dataset: h5.Dataset = self.field_data(name, arr.shape, arr.dtype)
            offsets: h5.Dataset = self.steps.field_data_offsets(name)
            sizes: h5.Dataset = self.steps.field_data_sizes(name)
            _append_offsets(dataset, offsets, arr)
            match arr.ndim:
                case 1:
                    _append(sizes, (1, arr.shape[0]))
                case 2:
                    _append(sizes, (arr.shape[1], arr.shape[0]))
                case _:
                    raise ValueError(arr.ndim)

    def append(self, mesh: pv.UnstructuredGrid, time: float) -> None:
        mesh._store_metadata()  # noqa: SLF001
        self._check_mesh(mesh)
        self._append_cells(mesh)
        self._append_points(mesh)
        self._append_point_data(mesh)
        self._append_cell_data(mesh)
        self._append_field_data(mesh)
        _append(self.steps.values, time)
        self.steps.n_steps += 1


def _append(dataset: h5.Dataset, delta: ArrayLike) -> None:
    delta: np.ndarray = np.asarray(delta)
    delta: np.ndarray = np.expand_dims(delta, tuple(range(dataset.ndim - delta.ndim)))
    if delta.size > 0:
        dataset.resize(dataset.len() + delta.shape[0], axis=0)
        dataset[-delta.shape[0] :] = delta


def _append_offsets(dataset: h5.Dataset, offsets: h5.Dataset, delta: ArrayLike) -> None:
    delta: np.ndarray = np.asarray(delta)
    delta: np.ndarray = np.expand_dims(delta, tuple(range(dataset.ndim - delta.ndim)))
    if _tail_equal(dataset, delta):
        offsets.resize(offsets.len() + 1, axis=0)
        offsets[-1] = dataset.len() - delta.shape[0]
    else:
        offset: int = dataset.len()
        if delta.size > 0:
            dataset.resize(dataset.len() + delta.shape[0], axis=0)
            dataset[-delta.shape[0] :] = delta
        offsets.resize(offsets.len() + 1, axis=0)
        offsets[-1] = offset


def _check_attributes(group: h5.Group, attributes: pv.DataSetAttributes) -> None:
    expected_keys: set[str] = set(group.keys())
    actual_keys: set[str] = set(attributes.keys())
    if expected_keys != actual_keys:
        msg: str = f"Attribute names mismatch for {attributes.association} data: expected {expected_keys}, got {actual_keys}"
        raise ValueError(msg)

    for name, arr_pv in attributes.items():
        arr: np.ndarray = _sanitize_array(arr_pv)
        dataset: h5.Dataset = group[name]

        if dataset.shape[1:] != arr.shape[1:]:
            msg: str = f"Shape mismatch for {attributes.association} data {name!r}: expected {dataset.maxshape}, got {arr.shape}"
            raise ValueError(msg)

        dataset_dtype: np.dtype = dataset.dtype
        if np.issubdtype(dataset_dtype, np.object_):
            dataset_dtype: np.dtype = np.dtype("T")
        if not np.issubdtype(arr.dtype, dataset_dtype):
            msg: str = f"Type mismatch for {attributes.association} data {name!r}: expected {dataset_dtype}, got {arr.dtype}"
            raise ValueError(msg)


def _sanitize_array(arr: np.ndarray | str) -> np.ndarray:
    if isinstance(arr, str):
        arr: np.ndarray = np.array([arr], "T")
    match arr.dtype.kind:
        case "b":
            arr: np.ndarray = arr.astype(np.uint8)
        case "U":
            arr: np.ndarray = arr.astype("T")
    return arr


def _tail_equal(dataset: h5.Dataset, delta: np.ndarray) -> bool:
    if dataset.len() == 0 or delta.size == 0:
        return False
    tail: np.ndarray = dataset[-delta.shape[0] :]
    equal_nan: bool = dataset.dtype.kind in {"b", "i", "u", "f", "c"}
    return np.array_equal(tail, delta, equal_nan=equal_nan)
