import subprocess
import tempfile
from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    cranium: Path = cherries.data("02-intermediate/cranium.vtp")
    mandible: Path = cherries.data("02-intermediate/mandible.vtp")
    skin: Path = cherries.data("02-intermediate/skin-with-mouth-socket.ply")

    output: Path = cherries.data("02-intermediate/tetgen.vtu")


def main(cfg: Config) -> None:
    cherries.log_input(cfg.cranium)
    cranium: pv.PolyData = melon.load_poly_data(cfg.cranium)
    cherries.log_input(cfg.mandible)
    mandible: pv.PolyData = melon.load_poly_data(cfg.mandible)
    cherries.log_input(cfg.skin)
    skin: pv.PolyData = melon.load_poly_data(cfg.skin)

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        cranium_file: Path = tmpdir / "cranium.ply"
        mandible_file: Path = tmpdir / "mandible.ply"
        skin_file: Path = tmpdir / "skin.ply"
        melon.save(cranium_file, cranium)
        melon.save(mandible_file, mandible)
        melon.save(skin_file, skin)
        csg: dict = {
            "operation": "difference",
            "left": str(skin_file),
            "right": {
                "operation": "union",
                "left": str(cranium_file),
                "right": str(mandible_file),
            },
        }
        csg_file: Path = tmpdir / "csg.json"
        grapes.save(csg_file, csg)
        output_file: Path = tmpdir / "output.msh"
        subprocess.run(
            ["fTetWild", "--output", output_file, "--csg", csg_file], check=True
        )
        output: pv.UnstructuredGrid = melon.load_unstructured_grid(output_file)
    melon.save(cfg.output, output)
    cherries.log_output(cfg.output)


if __name__ == "__main__":
    cherries.run(main)
