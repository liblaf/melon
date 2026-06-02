import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jinja2 as j2
import numpy as np
import trimesh as tm
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.melon import io

from ._template import get_environment

logger: logging.Logger = logging.getLogger(__name__)


def annotate_landmarks(
    left: Any,
    right: Any,
    *,
    left_landmarks: Float[ArrayLike, "L 3"] | None = None,
    right_landmarks: Float[ArrayLike, "L 3"] | None = None,
) -> tuple[Float[np.ndarray, "L 3"], Float[np.ndarray, "L 3"]]:
    left: tm.Trimesh = io.as_trimesh(left, triangulate=True)
    right: tm.Trimesh = io.as_trimesh(right, triangulate=True)
    if left_landmarks is None:
        left_landmarks: Float[np.ndarray, "L 3"] = np.zeros((0, 3))
    if right_landmarks is None:
        right_landmarks: Float[np.ndarray, "L 3"] = np.zeros((0, 3))

    environment: j2.Environment = get_environment()
    template: j2.Template = environment.get_template("annotate-landmarks.wrap")

    if np.size(left_landmarks) > 0 and np.shape(left_landmarks) == np.shape(
        right_landmarks
    ):
        transform, transformed, cost = tm.registration.procrustes(
            left_landmarks, right_landmarks
        )
        logger.info("procrustes cost: %f", cost)
        left: tm.Trimesh = left.copy()
        left: tm.Trimesh = left.apply_transform(transform)
        left_landmarks: Float[np.ndarray, "L 3"] = transformed
    else:
        transform, cost = tm.registration.mesh_other(left, right, scale=True)
        logger.info("mesh_other cost: %f", cost)
        left: tm.Trimesh = left.copy()
        left: tm.Trimesh = left.apply_transform(transform)
        left_landmarks: Float[np.ndarray, "L 3"] = tm.transform_points(
            left_landmarks, transform
        )

    with tempfile.TemporaryDirectory() as tmpdir_:
        tmpdir: Path = Path(tmpdir_)
        project_file: Path = tmpdir / "annotate-landmarks.wrap"
        left_file: Path = tmpdir / "left.obj"
        right_file: Path = tmpdir / "right.obj"
        left_landmarks_file: Path = tmpdir / "left.landmarks.json"
        right_landmarks_file: Path = tmpdir / "right.landmarks.json"
        io.save(left, left_file)
        io.save(right, right_file)
        io.save_landmarks(left_landmarks, left_landmarks_file)
        io.save_landmarks(right_landmarks, right_landmarks_file)
        project: str = template.render(
            {
                "left": left_file,
                "right": right_file,
                "left_landmarks": left_landmarks_file,
                "right_landmarks": right_landmarks_file,
            }
        )
        project_file.write_text(project)
        subprocess.run(["Wrap.sh", project_file], check=True)
        left_landmarks: Float[np.ndarray, "L 3"] = io.load_landmarks(
            left_landmarks_file
        )
        right_landmarks: Float[np.ndarray, "L 3"] = io.load_landmarks(
            right_landmarks_file
        )

    left_landmarks: Float[np.ndarray, "L 3"] = tm.transform_points(
        left_landmarks, tm.transformations.inverse_matrix(transform)
    )
    return left_landmarks, right_landmarks
