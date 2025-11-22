from typing import Any, no_type_check

import numpy as np
import pyvista as pv
import trimesh as tm
import warp as wp
from jaxtyping import Bool

from liblaf.melon import io


def contains(mesh: Any, pcl: Any) -> Bool[np.ndarray, " N"]:
    mesh_pv: pv.PolyData = io.as_polydata(mesh)
    mesh_wp: wp.Mesh = io.as_warp_mesh(mesh)
    mesh_tm: tm.Trimesh = io.as_trimesh(mesh)
    pcl: pv.PointSet = io.as_pointset(pcl)
    output_np: Bool[np.ndarray, " N"] = tm.bounds.contains(mesh_tm.bounds, pcl.points)
    points: wp.array = wp.from_numpy(pcl.points[output_np], dtype=wp.vec3f)
    max_dist: float = mesh_pv.length
    output: wp.array = wp.zeros(points.shape, dtype=wp.bool)
    wp.launch(
        _contains_kernel,
        dim=points.shape,
        inputs=[mesh_wp.id, points, max_dist],
        outputs=[output],
    )
    output_np[output_np] = output.numpy()
    return output_np


@wp.kernel
@no_type_check
def _contains_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    max_dist: wp.float32,
    # outputs
    output: wp.array(dtype=wp.bool),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point(mesh_id, point, max_dist)
    output[tid] = query.sign < 0
