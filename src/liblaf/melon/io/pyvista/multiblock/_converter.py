from collections.abc import Mapping
from typing import Any

import pyvista as pv
import trimesh as tm

from liblaf.melon.io.abc import ConverterDispatcher
from liblaf.melon.utils import filter_kwargs

as_multiblock: ConverterDispatcher[pv.MultiBlock] = ConverterDispatcher(pv.MultiBlock)


@as_multiblock.register(tm.Scene)
def _scene_to_multiblock(obj: tm.Scene, /, **kwargs) -> pv.MultiBlock:
    multiblock: pv.MultiBlock = pv.MultiBlock()
    wrap_kwargs: Mapping[str, Any] = filter_kwargs(pv.wrap, kwargs)
    for geometry_tm in obj.dump():
        geometry_pv: pv.DataSet = pv.wrap(geometry_tm, **wrap_kwargs)
        name: str = geometry_tm.metadata["name"]
        multiblock.append(geometry_pv, name=name)
    return multiblock
