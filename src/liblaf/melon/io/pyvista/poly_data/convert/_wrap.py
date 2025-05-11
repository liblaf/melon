from typing import Any

import pyvista as pv

from liblaf.melon import utils
from liblaf.melon.io.abc import AbstractConverter


class WrapToPolyData(AbstractConverter):
    def match_from(self, data: Any) -> bool:
        return utils.is_trimesh(data) or utils.is_numpy(data)

    def convert(self, data: Any, **kwargs) -> Any:
        return pv.wrap(data, **kwargs)
