from pathlib import Path

import pytetwild
import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    input: Path = Path("data/02_intermediate/")
    output: Path = Path("data/02_intermediate/")


@cherries.main()
def main(cfg: Config) -> None:
    cranium: pv.PolyData = melon.load_poly_data(cfg.input / "cranium.ply")
    mandible: pv.PolyData = melon.load_poly_data(cfg.input / "mandible.ply")
    skin: pv.PolyData = melon.load_poly_data(cfg.input / "skin.ply")
    skull: pv.PolyData = pv.merge([cranium, mandible])
    skull.triangulate(inplace=True)
    skull.flip_normals()
    surface: pv.PolyData = pv.merge([skin, skull])
    melon.save(cfg.output / "surface.ply", surface)
    mesh: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(surface)
    melon.save(cfg.output / "tetgen.vtu", mesh)


if __name__ == "__main__":
    main(Config())
