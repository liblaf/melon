from typing import Any

import pyvista as pv

from liblaf.melon.io.abc import UnsupportedConversionError, conversion_dispatcher
from liblaf.melon.io.pyvista.poly_data import as_poly_data

from ._poly_data import PolyDataToPointSet

conversion_dispatcher.register(PolyDataToPointSet())


def as_point_set(data: Any, *, point_normals: bool = False) -> pv.PointSet:
    if point_normals:
        return _as_point_set_with_normals(data)
    return _as_point_set(data)


def _as_point_set(data: Any) -> pv.PointSet:
    return conversion_dispatcher.convert(data, pv.PointSet)


def _as_point_set_with_normals(data: Any) -> pv.PointSet:
    try:
        mesh: pv.PolyData = as_poly_data(data)
    except UnsupportedConversionError:
        pass
    else:
        if mesh.active_scalars_info.association != pv.FieldAssociation.POINT:
            mesh.set_active_scalars(None)
        data: pv.PointSet = mesh.cast_to_pointset()
        data.point_data["Normals"] = mesh.point_normals
        return data
    data: pv.PointSet = _as_point_set(data)
    if "Normals" in data.point_data:
        return data
    # TODO: estimate normals
    raise NotImplementedError
