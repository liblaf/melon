from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Integer


def load_polydata_obj(path: Path, **_kwargs) -> pv.PolyData:
    mesh: pv.PolyData = pv.read(path)  # pyright: ignore[reportAssignmentType]
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
