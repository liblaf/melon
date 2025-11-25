from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    cranium: Path = cherries.input("14-cranium.ply")
    mandible: Path = cherries.input("14-mandible.ply")
    muscles: Path = cherries.input("20-muscles.vtm")
    skin: Path = cherries.input("12-skin.vtp")

    output: Path = cherries.output("21-tetgen.vtu")

    lr: float = 0.05 * 0.5
    epsr: float = 1e-3 * 0.5


def main(cfg: Config) -> None:
    cranium: pv.PolyData = melon.load_polydata(cfg.cranium)
    mandible: pv.PolyData = melon.load_polydata(cfg.mandible)
    skin: pv.PolyData = melon.load_polydata(cfg.skin)

    # fix mouth region
    # lip_top: pv.PolyData = melon.tri.extract_groups(skin, ["LipTop", "LipInnerTop"])
    lip_bottom: pv.PolyData = melon.tri.extract_groups(
        skin, ["LipBottom", "LipInnerBottom"]
    )
    cranium.remove_points(
        melon.bounds_contains(lip_bottom.bounds, cranium.points), inplace=True
    )
    melon.save(cherries.temp("21-cranium-clip.vtp"), cranium)

    skull: pv.PolyData = pv.merge([cranium, mandible])
    muscles: pv.MultiBlock = pv.read(cfg.muscles)  # pyright: ignore[reportAssignmentType]
    muscles_combine: pv.PolyData = muscles.combine().extract_surface()

    tetmesh: pv.UnstructuredGrid = melon.tetwild(
        pv.merge([muscles_combine, skin, skull.flip_faces()]), lr=cfg.lr, epsr=cfg.epsr
    )
    tetmesh.cell_data.clear()
    ic(tetmesh.n_points)
    ic(tetmesh.n_cells)
    melon.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.main(main)
