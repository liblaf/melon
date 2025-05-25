from collections.abc import Container
from typing import override

import pyvista as pv

from liblaf.melon.io.abc import AbstractReader
from liblaf.melon.typed import PathLike


class ObjReader(AbstractReader):
    extensions: Container[str] = {".obj"}
    precedence: int = 1  # prefer over pyvista.read

    @override
    def load(self, path: PathLike, /, **kwargs) -> pv.PolyData:
        import numpy as np
        import tinyobjloader
        from jaxtyping import Float

        reader = tinyobjloader.ObjReader()
        ok: bool = reader.ParseFromFile(str(path))
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
