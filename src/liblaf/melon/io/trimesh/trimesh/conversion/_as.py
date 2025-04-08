from typing import Any

import trimesh as tm

from liblaf.melon.io import abc

from ._poly_data import PolyDataToTrimesh

abc.register_converter(PolyDataToTrimesh())


def as_trimesh(data: Any) -> tm.Trimesh:
    return abc.convert(data, tm.Trimesh)
