import functools
from collections.abc import Generator
from pathlib import Path
from types import TracebackType
from typing import Self, SupportsFloat

import attrs
import h5py as h5
import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike

DEFAULT_DATASET_OPTIONS: dict[str, object] = {
    "exact": True,
    "compression": "gzip",
    "compression_opts": 4,
    "shuffle": True,
}


def _sanitize_name(name: str) -> str:
    for invalid_char in ["/", "."]:
        if invalid_char in name:
            raise ValueError(name)
    return name


def _sanitize_array(arr: ArrayLike | str) -> np.ndarray:
    if isinstance(arr, str):
        arr: np.ndarray = np.array([arr], "T")
    arr: np.ndarray = np.atleast_1d(arr)
    match arr.dtype.kind:
        case "b":
            arr: np.ndarray = arr.astype(np.uint8)
        case "O":
            arr: np.ndarray = arr.astype("T")
        case "U":
            arr: np.ndarray = arr.astype("T")
    return arr


class FileWrapper:
    file: h5.File

    def group(self, name: str) -> h5.Group:
        return self.file.require_group(name)

    def dataset(self, name: str, data: ArrayLike | None = None) -> h5.Dataset:
        if data is None:
            return self.file[name]
        data: np.ndarray = _sanitize_array(data)
        kwargs: dict[str, object] = DEFAULT_DATASET_OPTIONS.copy()
        if data.size > 0:
            kwargs["chunks"] = data.shape
        dataset: h5.Dataset = self.file.require_dataset(
            name=name,
            shape=(0, *data.shape[1:]),
            dtype=data.dtype,
            maxshape=(None, *data.shape[1:]),
            **kwargs,
        )
        if data.size > 0:
            dataset.resize(dataset.len() + data.shape[0], axis=0)
            dataset[-data.shape[0] :] = data
        return dataset

    def dataset_offsets(
        self, dataset_name: str, offsets_name: str, data: ArrayLike
    ) -> tuple[h5.Dataset, h5.Dataset]:
        data: np.ndarray = _sanitize_array(data)
        if not self.dataset_tail_equal(dataset_name, data):
            dataset: h5.Dataset = self.dataset(dataset_name, data)
        else:
            dataset: h5.Dataset = self.dataset(dataset_name)
        offsets: h5.Dataset = self.dataset(
            offsets_name, np.array([dataset.len() - data.shape[0]])
        )
        return dataset, offsets

    def dataset_tail_equal(self, name: str, data: ArrayLike) -> bool:
        data: np.ndarray = _sanitize_array(data)
        if name not in self.file:
            return False
        dataset: h5.Dataset = self.file[name]
        if dataset.len() == 0 or data.size == 0:
            return False
        tail: np.ndarray = dataset[-data.shape[0] :]
        tail: np.ndarray = _sanitize_array(tail)
        if tail.dtype != data.dtype:
            return False
        numeric: bool = dataset.dtype.kind in {"b", "i", "u", "f", "c"}
        return np.array_equal(tail, data, equal_nan=numeric)


@attrs.define
class StepsGroup(FileWrapper):
    file: h5.File

    @property
    def n_steps(self) -> int:
        group: h5.Group = self.group("/VTKHDF/Steps")
        return group.attrs["NSteps"]

    @n_steps.setter
    def n_steps(self, value: int) -> None:
        group: h5.Group = self.group("/VTKHDF/Steps")
        group.attrs["NSteps"] = value


@attrs.define
class VTKHDFTemporalUnstructuredGridWriter(FileWrapper):
    _file: Path = attrs.field(converter=Path, alias="file")

    def __enter__(self) -> Self:
        self.file.__enter__()
        group: h5.Group = self.group("/VTKHDF")
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

    @functools.cached_property
    def file(self) -> h5.File:
        return h5.File(self._file, "w")

    @functools.cached_property
    def steps(self) -> StepsGroup:
        return StepsGroup(self.file)

    @property
    def n_steps(self) -> int:
        return self.steps.n_steps

    @n_steps.setter
    def n_steps(self, value: int) -> None:
        self.steps.n_steps = value

    def _append_topology(self, mesh: pv.UnstructuredGrid) -> None:
        if not (
            self.dataset_tail_equal("/VTKHDF/NumberOfPoints", mesh.n_points)
            and self.dataset_tail_equal(
                "/VTKHDF/NumberOfConnectivityIds", mesh.cell_connectivity.size
            )
            and self.dataset_tail_equal("/VTKHDF/NumberOfCells", mesh.n_cells)
            and self.dataset_tail_equal("/VTKHDF/Types", mesh.celltypes)
            and self.dataset_tail_equal("/VTKHDF/Offsets", mesh.offset)
            and self.dataset_tail_equal("/VTKHDF/Connectivity", mesh.cell_connectivity)
        ):
            self.dataset("/VTKHDF/NumberOfPoints", mesh.n_points)
            self.dataset("/VTKHDF/NumberOfConnectivityIds", mesh.cell_connectivity.size)
            self.dataset("/VTKHDF/NumberOfCells", mesh.n_cells)
            self.dataset("/VTKHDF/Types", mesh.celltypes)
            self.dataset("/VTKHDF/Offsets", mesh.offset)
            self.dataset("/VTKHDF/Connectivity", mesh.cell_connectivity)
        self.dataset(
            "/VTKHDF/Steps/PartOffsets",
            self.dataset("/VTKHDF/NumberOfPoints").len() - 1,
        )
        self.dataset("/VTKHDF/Steps/NumberOfParts", 1)
        self.dataset(
            "/VTKHDF/Steps/CellOffsets",
            self.dataset("/VTKHDF/Types").len() - mesh.n_cells,
        )
        self.dataset(
            "/VTKHDF/Steps/ConnectivityIdOffsets",
            self.dataset("/VTKHDF/Connectivity").len() - mesh.cell_connectivity.size,
        )

    def _append_points(self, mesh: pv.UnstructuredGrid) -> None:
        self.dataset_offsets(
            "/VTKHDF/Points", "/VTKHDF/Steps/PointOffsets", mesh.points
        )

    def _check_attributes(
        self, group_name: str, attributes: pv.DataSetAttributes
    ) -> None:
        if self.n_steps == 0:
            return
        if group_name in self.file:
            group: h5.Group = self.group(group_name)
            expected: frozenset[str] = frozenset(group.keys())
        else:
            expected: frozenset[str] = frozenset()
        actual: frozenset[str] = frozenset(_sanitize_name(name) for name in attributes)
        if expected != actual:
            msg: str = f"missing: {expected - actual}, extra: {actual - expected}"
            raise ValueError(msg)

    @staticmethod
    def _iter_attributes(
        attributes: pv.DataSetAttributes,
    ) -> Generator[tuple[str, np.ndarray]]:
        for name, arr in attributes.items():
            name: str = _sanitize_name(name)  # noqa: PLW2901
            arr: np.ndarray = _sanitize_array(arr)  # noqa: PLW2901
            yield name, arr

    def _append_point_data(self, mesh: pv.UnstructuredGrid) -> None:
        self._check_attributes("/VTKHDF/PointData", mesh.point_data)
        for name, arr in self._iter_attributes(mesh.point_data):
            self.dataset_offsets(
                f"/VTKHDF/PointData/{name}",
                f"/VTKHDF/Steps/PointDataOffsets/{name}",
                arr,
            )

    def _append_cell_data(self, mesh: pv.UnstructuredGrid) -> None:
        self._check_attributes("/VTKHDF/CellData", mesh.cell_data)
        for name, arr in self._iter_attributes(mesh.cell_data):
            self.dataset_offsets(
                f"/VTKHDF/CellData/{name}", f"/VTKHDF/Steps/CellDataOffsets/{name}", arr
            )

    def _append_field_data(self, mesh: pv.UnstructuredGrid) -> None:
        self._check_attributes("/VTKHDF/FieldData", mesh.field_data)
        for name, arr in self._iter_attributes(mesh.field_data):
            self.dataset_offsets(
                f"/VTKHDF/FieldData/{name}",
                f"/VTKHDF/Steps/FieldDataOffsets/{name}",
                arr,
            )
            match arr.shape:
                case (n_tuples,):
                    size: tuple[int, int] = (1, n_tuples)
                case (n_tuples, n_components):
                    size: tuple[int, int] = (n_components, n_tuples)
            self.dataset(f"/VTKHDF/Steps/FieldDataSizes/{name}", np.atleast_2d(size))

    def append(
        self, mesh: pv.UnstructuredGrid, time: SupportsFloat | None = None
    ) -> None:
        if time is None:
            if "/VTKHDF/Steps/Values" in self.file:
                time: float = self.dataset("/VTKHDF/Steps/Values")[-1] + 1.0
            else:
                time: float = 0.0
        else:
            time: float = float(time)
        mesh._store_metadata()  # noqa: SLF001
        self._append_topology(mesh)
        self._append_points(mesh)
        self._append_point_data(mesh)
        self._append_cell_data(mesh)
        self._append_field_data(mesh)
        self.dataset("/VTKHDF/Steps/Values", time)
        self.steps.n_steps += 1
