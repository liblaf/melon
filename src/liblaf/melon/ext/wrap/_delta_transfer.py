import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jinja2 as j2
import pyvista as pv

from liblaf.melon import io

from ._template import get_environment


def delta_transfer(
    floating_neutral: pv.PolyData, ref_neutral: Any, ref_expression: Any
) -> pv.PolyData:
    environment: j2.Environment = get_environment()
    template: j2.Template = environment.get_template("delta-transfer.wrap")

    with tempfile.TemporaryDirectory() as tmpdir_:
        tmpdir: Path = Path(tmpdir_)
        project_path: Path = tmpdir / "delta-transfer.wrap"
        floating_neutral_path: Path = tmpdir / "floating-neutral.obj"
        ref_neutral_path: Path = tmpdir / "ref-neutral.obj"
        ref_expression_path: Path = tmpdir / "ref-expression.obj"
        output_path: Path = tmpdir / "output.obj"
        io.save(floating_neutral, floating_neutral_path)
        io.save(ref_neutral, ref_neutral_path)
        io.save(ref_expression, ref_expression_path)
        project: str = template.render(
            {
                "floating_neutral": floating_neutral_path,
                "ref_neutral": ref_neutral_path,
                "ref_expression": ref_expression_path,
                "output": output_path,
            }
        )
        project_path.write_text(project)
        subprocess.run(["WrapCmd.sh", "compute", project_path], check=True)
        output: pv.PolyData = io.load_polydata(output_path)
        output.copy_attributes(floating_neutral)
    return output
