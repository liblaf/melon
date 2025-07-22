from typing import Any

import trimesh as tm

from liblaf.melon import io


def is_volume(mesh: Any) -> bool:
    mesh: tm.Trimesh = io.as_trimesh(mesh)
    return mesh.is_volume
