import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jinja2 as j2
import numpy as np
from jaxtyping import Integer
from numpy.typing import ArrayLike

from liblaf.melon import io

from ._template import get_environment


def select_polygons(
    mesh: Any, polygons: Integer[ArrayLike, " N"] | None = None
) -> Integer[np.ndarray, " N"]:
    if polygons is None:
        polygons: Integer[np.ndarray, " N"] = np.empty((0,), np.int32)

    environment: j2.Environment = get_environment()
    template: j2.Template = environment.get_template("select-polygons.wrap")

    with tempfile.TemporaryDirectory() as tmpdir_:
        tmpdir: Path = Path(tmpdir_)
        project_path: Path = tmpdir / "select-polygons.wrap"
        mesh_path: Path = tmpdir / "mesh.obj"
        polygons_path: Path = tmpdir / "polygons.json"
        io.save(mesh, mesh_path)
        io.save_polygons(polygons, polygons_path)
        project: str = template.render(
            {
                "mesh": mesh_path,
                "polygons": polygons_path,
            }
        )
        project_path.write_text(project)
        subprocess.run(["Wrap.sh", project_path], check=True)
        polygons: Integer[np.ndarray, " N"] = io.load_polygons(polygons_path)
    return polygons
