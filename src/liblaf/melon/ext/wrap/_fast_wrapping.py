import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jinja2 as j2
import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.melon import io

from ._template import get_environment

logger: logging.Logger = logging.getLogger(__name__)


def fast_wrapping(
    floating: Any,
    fixed: Any,
    *,
    floating_landmarks: Float[ArrayLike, "L 3"] | None = None,
    fixed_landmarks: Float[ArrayLike, "L 3"] | None = None,
    free_polygons_floating: Integer[ArrayLike, " F"] | None = None,
) -> pv.PolyData:
    if floating_landmarks is None:
        floating_landmarks: Float[np.ndarray, "L 3"] = np.empty((0, 3))
    if fixed_landmarks is None:
        fixed_landmarks: Float[np.ndarray, "L 3"] = np.empty((0, 3))
    if free_polygons_floating is None:
        free_polygons_floating: Integer[np.ndarray, " F"] = np.empty((0,), dtype=int)

    environment: j2.Environment = get_environment()
    template: j2.Template = environment.get_template("fast-wrapping.wrap")

    floating: pv.PolyData = io.as_polydata(floating)
    if np.size(floating_landmarks):
        matrix, transformed, cost = tm.registration.procrustes(
            floating_landmarks, fixed_landmarks
        )
        logger.info("procrustes cost: %f", cost)
        floating: pv.PolyData = floating.transform(matrix, inplace=False)
        floating_landmarks: Float[np.ndarray, "L 3"] = transformed

    with tempfile.TemporaryDirectory() as tmpdir_:
        tmpdir: Path = Path(tmpdir_)
        project_file: Path = tmpdir / "fast-wrapping.wrap"
        floating_path: Path = tmpdir / "floating.obj"
        fixed_path: Path = tmpdir / "fixed.obj"
        floating_landmarks_path: Path = tmpdir / "floating-landmarks.json"
        fixed_landmarks_path: Path = tmpdir / "fixed-landmarks.json"
        free_polygons_floating_path: Path = tmpdir / "free-polygons-floating.json"
        output_path: Path = tmpdir / "output.obj"
        io.save(floating, floating_path)
        io.save(fixed, fixed_path)
        io.save_landmarks(floating_landmarks, floating_landmarks_path)
        io.save_landmarks(fixed_landmarks, fixed_landmarks_path)
        io.save_polygons(free_polygons_floating, free_polygons_floating_path)
        project: str = template.render(
            {
                "floating": floating_path,
                "fixed": fixed_path,
                "floating_landmarks": floating_landmarks_path,
                "fixed_landmarks": fixed_landmarks_path,
                "free_polygons_floating": free_polygons_floating_path,
                "output": output_path,
            }
        )
        project_file.write_text(project)
        subprocess.run(["WrapCmd.sh", "compute", project_file], check=True)
        result: pv.PolyData = io.load_polydata(output_path)

    result.copy_attributes(floating)
    return result
