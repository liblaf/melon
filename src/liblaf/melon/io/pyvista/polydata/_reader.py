from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pyvista as pv

from liblaf.melon.io.abc import ReaderDispatcher
from liblaf.melon.utils import filter_kwargs

load_polydata: ReaderDispatcher[pv.PolyData] = ReaderDispatcher(pv.PolyData)


@load_polydata.register_fallback
def load_polydata(path: Path, /, **kwargs) -> pv.PolyData:
    kwargs: Mapping[str, Any] = filter_kwargs(pv.read, kwargs)
    return cast("pv.PolyData", pv.read(path, cls=pv.PolyData, **kwargs))
