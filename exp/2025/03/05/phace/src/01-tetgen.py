from pathlib import Path

import pytetwild
import pyvista as pv

import liblaf.cherries as cherries  # noqa: PLR0402
import liblaf.melon as melon  # noqa: PLR0402


class Config(cherries.BaseConfig):
    raw: Path = Path("./data/00-raw/")
    intermediate: Path = Path("./data/01-intermediate/")


@cherries.main()
def main(cfg: Config) -> None:
    face: pv.PolyData = melon.load_poly_data(cfg.raw / "face.ply")
    cranium: pv.PolyData = melon.load_poly_data(cfg.raw / "cranium.ply")
    mandible: pv.PolyData = melon.load_poly_data(cfg.raw / "mandible.ply")
    skull: pv.PolyData = pv.merge([cranium, mandible])
    skull.flip_normals()
    surface: pv.PolyData = pv.merge([face, skull])
    mesh: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(surface)
    melon.save(cfg.intermediate / "00-tetgen.vtu", mesh)


if __name__ == "__main__":
    main(Config())
