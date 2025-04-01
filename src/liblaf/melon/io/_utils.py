from typing import Any

import pyvista as pv

TYPES_TO_EXT: list[tuple[type, str]] = [
    (pv.PolyData, ".vtp"),
    (pv.UnstructuredGrid, ".vtu"),
]


def identify_data_format(data: Any) -> str:
    for cls, ext in TYPES_TO_EXT:
        if isinstance(data, cls):
            return ext
    msg: str = f"Cannot identify data format for `{data}`."
    raise TypeError(msg)
