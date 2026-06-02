from typing import Any

import trimesh as tm

from liblaf.melon import io


def fix_normals(mesh: Any, *, multibody: bool | None = None) -> tm.Trimesh:
    mesh: tm.Trimesh = io.as_trimesh(mesh)
    mesh.fix_normals(multibody=multibody)
    return mesh
