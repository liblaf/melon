from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    cranium: Path = cherries.input("14-cranium.ply")
    mandible: Path = cherries.input("14-mandible.ply")
    muscles: Path = cherries.input("20-muscles.vtm")
    skin: Path = cherries.input("12-skin.vtp")


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

    # I tried many combinations of tetwild parameters, and got the following magic numbers
    def tetgen(
        *,
        coarsen: bool = False,
        conform: bool = False,
        epsr: float = 5e-4,
        lr: float = 0.05,
    ) -> pv.UnstructuredGrid:
        surface: pv.PolyData
        if conform:
            surface = pv.merge([skin, skull.flip_faces(), muscles_combine])
        else:
            surface = pv.merge([skin, skull.flip_faces()])
        result: pv.UnstructuredGrid = melon.tetwild(
            surface, lr=lr, epsr=epsr, coarsen=coarsen
        )
        filename: str = f"21-tetgen-{round(result.n_cells / 1000)}k"
        if coarsen:
            filename += "-coarse"
        if conform:
            filename += "-conform"
        ic(coarsen, conform, lr, epsr, result.n_cells)
        surface: pv.PolyData = result.extract_surface()  # pyright: ignore[reportAssignmentType]
        ic(melon.compute_edges_length(surface).mean() / surface.length)
        melon.save(cherries.output(f"{filename}.vtu"), result)
        return result

    tetgen(coarsen=True, conform=False, epsr=4e-4, lr=0.05)
    tetgen(coarsen=False, conform=False, epsr=5e-4, lr=0.05)
    # tetgen(coarsen=False, conform=False, lr=0.03)
    # tetgen(coarsen=False, conform=False, lr=0.02)
    tetgen(coarsen=False, conform=False, epsr=3e-4, lr=0.015)
    tetgen(coarsen=False, conform=False, epsr=2e-4, lr=0.01)
    tetgen(coarsen=False, conform=False, epsr=1e-4, lr=0.005)

    # tetgen(coarsen=True, conform=True, epsr=4e-4, lr=0.05)
    # tetgen(coarsen=False, conform=True, lr=0.05)
    # tetgen(coarsen=False, conform=True, lr=0.03)
    # tetgen(coarsen=False, conform=True, lr=0.02)
    # tetgen(coarsen=False, conform=True, lr=0.01)


if __name__ == "__main__":
    cherries.main(main)
