from collections.abc import Container
from pathlib import Path
from typing import override

import numpy as np
import pyvista as pv
from jaxtyping import Integer

from liblaf.melon.io.abc import AbstractReader
from liblaf.melon.typed import PathLike


class ObjReader(AbstractReader):
    extensions: Container[str] = {".obj"}
    precedence: int = 1  # prefer over pyvista.read

    @override
    def load(self, path: PathLike, /, **kwargs) -> pv.PolyData:
        mesh: pv.PolyData = self._load(path)
        if kwargs.pop("clean", True):
            mesh.clean(inplace=True)
        return mesh

    def _load(self, path: PathLike, /) -> pv.PolyData:
        path: Path = Path(path)
        mesh: pv.PolyData = pv.read(path)
        if "GroupIds" not in mesh.cell_data:
            return mesh
        group_id_to_name: list[str] = []
        group_name_to_id: dict[str, int] = {}
        old_id_to_new_id: list[int] = []
        with path.open() as fp:
            for line in fp:
                tokens: list[str] = line.split(maxsplit=1)
                if not tokens or tokens[0] != "g":
                    continue
                group_name: str
                if len(tokens) == 1:
                    group_name = f"Group_{len(group_id_to_name):03d}"
                else:
                    group_name = tokens[1].strip()
                group_id: int
                if group_name not in group_name_to_id:
                    assert len(group_id_to_name) == len(group_name_to_id)
                    group_id = len(group_name_to_id)
                    group_name_to_id[group_name] = group_id
                    group_id_to_name.append(group_name)
                else:
                    group_id = group_name_to_id[group_name]
                old_id_to_new_id.append(group_id)
        old_ids: Integer[np.ndarray, " C"] = np.asarray(
            mesh.cell_data["GroupIds"], dtype=int
        )
        if np.all(old_ids == 0) and len(group_id_to_name) == 0:
            return mesh
        new_ids: Integer[np.ndarray, " C"] = np.asarray(old_id_to_new_id)[old_ids]
        mesh.cell_data["GroupIds"] = new_ids
        mesh.field_data["GroupNames"] = group_id_to_name
        return mesh


def _load_tinyobjloader(path: PathLike, /) -> pv.PolyData:
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
        faces.extend(_as_cell_array(mesh.num_face_vertices, mesh.vertex_indices()))
        group_ids.extend([group_id] * len(mesh.num_face_vertices))
        group_names.append(shape.name)
    data = pv.PolyData(vertices, faces=faces)
    data.cell_data["GroupIds"] = group_ids
    data.field_data["GroupNames"] = group_names
    return data


def _as_cell_array(
    num_face_vertices: list[int], vertex_indices: list[int]
) -> list[int]:
    faces: list[int] = []
    index_offset: int = 0
    for fv in num_face_vertices:
        faces.append(fv)
        faces.extend(vertex_indices[index_offset : index_offset + fv])
        index_offset += fv
    return faces
