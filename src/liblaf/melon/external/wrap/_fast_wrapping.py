import os
import subprocess as sp
import tempfile
from pathlib import Path
from typing import Any

import jinja2
import pyvista as pv
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.melon import io

from ._templates import environment


def fast_wrapping(
    source: Any,
    target: Any,
    *,
    source_landmarks: Float[ArrayLike, "L 3"] | None = None,
    target_landmarks: Float[ArrayLike, "L 3"] | None = None,
    free_polygons_floating: Integer[ArrayLike, " F"] | None = None,
    verbose: bool = True,
) -> pv.PolyData:
    source_landmarks = source_landmarks if source_landmarks is not None else []
    target_landmarks = target_landmarks if target_landmarks is not None else []
    free_polygons_floating = (
        free_polygons_floating if free_polygons_floating is not None else []
    )
    with tempfile.TemporaryDirectory(delete=False) as tmpdir_str:
        tmpdir: Path = Path(tmpdir_str).absolute()
        project_file: Path = tmpdir / "fast-wrapping.wrap"
        source_file: Path = tmpdir / "source.obj"
        target_file: Path = tmpdir / "target.obj"
        output_file: Path = tmpdir / "output.obj"
        source_landmarks_file: Path = tmpdir / "source.landmarks.json"
        target_landmarks_file: Path = tmpdir / "target.landmarks.json"
        free_polygons_floating_file: Path = tmpdir / "free-polygons-floating.json"
        io.save(source_file, source)
        io.save(target_file, target)
        io.save_landmarks(source_landmarks_file, source_landmarks)
        io.save_landmarks(target_landmarks_file, target_landmarks)
        io.save_polygons(free_polygons_floating_file, free_polygons_floating)
        template: jinja2.Template = environment.get_template("fast-wrapping.wrap")
        project: str = template.render(
            {
                "source": str(source_file),
                "target": str(target_file),
                "output": str(output_file),
                "source_landmarks": str(source_landmarks_file),
                "target_landmarks": str(target_landmarks_file),
                "free_polygons_floating": str(free_polygons_floating_file),
            }
        )
        project_file.write_text(project)
        args: list[str | os.PathLike] = ["WrapCmd.sh", "compute", project_file]
        if verbose:
            args.append("--verbose")
        sp.run(args, check=True)
        return io.load_polydata(output_file)
