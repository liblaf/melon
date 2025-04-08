from typing import Any

import pyvista as pv

from liblaf.melon.io import abc


def as_image_data(data: Any) -> pv.ImageData:
    return abc.convert(data, pv.ImageData)
