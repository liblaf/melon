from typing import Any

import trimesh as tm

from liblaf import melon


def is_volume(mesh: Any) -> bool:
    mesh: tm.Trimesh = melon.as_trimesh(mesh)
    return mesh.is_volume
