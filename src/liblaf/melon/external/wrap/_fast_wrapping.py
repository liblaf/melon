import importlib.resources
import os
import string
import subprocess as sp
import tempfile
from pathlib import Path
from typing import Any

import pyvista as pv
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.melon import io


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
        template = string.Template(
            (
                importlib.resources.files("liblaf.melon.external.wrap")
                / "fast-wrapping.wrap"
            ).read_text()
        )
        project: str = template.substitute(
            {
                "SOURCE_FILE": str(source_file),
                "TARGET_FILE": str(target_file),
                "OUTPUT_FILE": str(output_file),
                "SOURCE_LANDMARKS_FILE": str(source_landmarks_file),
                "TARGET_LANDMARKS_FILE": str(target_landmarks_file),
                "FREE_POLYGONS_FLOATING_FILE": str(free_polygons_floating_file),
            }
        )
        project_file.write_text(project)
        args: list[str | os.PathLike] = ["WrapCmd.sh", "compute", project_file]
        if verbose:
            args.append("--verbose")
        sp.run(args, check=True)
        return io.load_poly_data(output_file)
