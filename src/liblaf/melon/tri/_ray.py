from typing import Any

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Bool

from liblaf.melon import io


def contains(mesh: Any, pcl: Any) -> Bool[np.ndarray, " N"]:
    mesh: tm.Trimesh = io.as_trimesh(mesh)
    pcl: pv.PointSet = io.as_pointset(pcl)
    return mesh.contains(pcl.points)
