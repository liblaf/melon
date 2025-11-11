import subprocess as sp
import tempfile
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.melon import io

from ._templates import environment


def annotate_landmarks(
    left: Any,
    right: Any,
    *,
    left_landmarks: Float[ArrayLike, "L 3"] | None = None,
    right_landmarks: Float[ArrayLike, "L 3"] | None = None,
) -> tuple[Float[np.ndarray, "L 3"], Float[np.ndarray, "L 3"]]:
    if left_landmarks is None:
        left_landmarks = np.zeros((0, 3))
    if right_landmarks is None:
        right_landmarks = np.zeros((0, 3))
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir: Path = Path(tmpdir_str).absolute()
        project_file: Path = tmpdir / "annotate-landmarks.wrap"
        left_file: Path = tmpdir / "left.obj"
        right_file: Path = tmpdir / "right.obj"
        left_landmarks_file: Path = tmpdir / "left.landmarks.json"
        right_landmarks_file: Path = tmpdir / "right.landmarks.json"
        io.save(left_file, left)
        io.save(right_file, right)
        io.save_landmarks(left_landmarks_file, left_landmarks)
        io.save_landmarks(right_landmarks_file, right_landmarks)
        template: jinja2.Template = environment.get_template("annotate-landmarks.wrap")
        project: str = template.render(
            {
                "left": left_file,
                "right": right_file,
                "left_landmarks": left_landmarks_file,
                "right_landmarks": right_landmarks_file,
            }
        )
        project_file.write_text(project)
        sp.run(["Wrap.sh", project_file], check=True)
        left_landmarks = io.load_landmarks(left_landmarks_file)
        right_landmarks = io.load_landmarks(right_landmarks_file)
    return left_landmarks, right_landmarks
