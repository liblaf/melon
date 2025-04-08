import os
from pathlib import Path

import numpy as np
import pyvista as pv
import tinyobjloader
from jaxtyping import Float

from liblaf import grapes


def load_obj(fpath: str | os.PathLike[str]) -> pv.PolyData:
    fpath: Path = grapes.as_path(fpath)
    reader = tinyobjloader.ObjReader()
    ok: bool = reader.ParseFromFile(str(fpath))
    if not ok:
        raise RuntimeError(reader.Error())
    attrib: tinyobjloader.attrib_t = reader.GetAttrib()
    vertices: Float[np.ndarray, "V 3"] = np.asarray(attrib.vertices).reshape(-1, 3)
    shapes: list[tinyobjloader.shape_t] = reader.GetShapes()
    faces: list[int] = []
    group_ids: list[int] = []
    group_names: list[str] = []
    for group_id, shape in enumerate(shapes):
        mesh: tinyobjloader.mesh_t = shape.mesh
        faces.extend(as_cell_array(mesh.num_face_vertices, mesh.vertex_indices()))
        group_ids.extend([group_id] * len(mesh.num_face_vertices))
        group_names.append(shape.name)
    data = pv.PolyData(vertices, faces=faces)
    data.cell_data["GroupIds"] = group_ids
    data.field_data["GroupNames"] = group_names
    return data


def as_cell_array(num_face_vertices: list[int], vertex_indices: list[int]) -> list[int]:
    faces: list[int] = []
    index_offset: int = 0
    for fv in num_face_vertices:
        faces.append(fv)
        faces.extend(vertex_indices[index_offset : index_offset + fv])
        index_offset += fv
    return faces
