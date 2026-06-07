from typing import Any

import trimesh as tm

from liblaf.melon import io


def fix_normals(mesh: Any, *, multibody: bool | None = None) -> tm.Trimesh:
    """Orient triangle normals with Trimesh and return a Trimesh object.

    Args:
        mesh: Object convertible to [`trimesh.Trimesh`][trimesh.Trimesh].
        multibody: Forwarded to [`trimesh.Trimesh.fix_normals`][].

    Returns:
        Mesh with repaired normal orientation.
    """
    mesh: tm.Trimesh = io.as_trimesh(mesh)
    mesh.fix_normals(multibody=multibody)
    return mesh
