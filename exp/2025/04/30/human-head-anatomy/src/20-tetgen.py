from collections.abc import Mapping
from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("01-raw/Full human head anatomy.obj")
    groups: Path = cherries.input("02-intermediate/groups.toml")
    skin: Path = cherries.input("02-intermediate/skin-with-mouth-socket.ply")

    output: Path = cherries.output("02-intermediate/20-tetgen.vtu")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    skin: pv.PolyData = melon.load_poly_data(cfg.skin)
    brain: pv.PolyData = melon.triangle.extract_groups(full, groups["Brain"])
    nervous: pv.PolyData = melon.triangle.extract_groups(full, groups["Nervous"])
    skull: pv.PolyData = melon.triangle.extract_groups(
        full, groups["cranium"] + groups["mandible"]
    )

    tetmesh: pv.UnstructuredGrid = melon.tetwild(
        {
            "operation": "difference",
            "left": skin,
            "right": csg_union(brain, nervous, skull),
        },
        lr=0.05 * 0.25,
        epsr=1e-3 * 0.25,
        csg=True,
    )
    cherries.log_metric("n_points", tetmesh.n_points)
    cherries.log_metric("n_cells", tetmesh.n_cells)

    melon.save(cfg.output, tetmesh)


def csg_union(left: pv.PolyData, *geometries: pv.PolyData) -> Mapping | pv.PolyData:
    if not geometries:
        return left
    return {"operation": "union", "left": left, "right": csg_union(*geometries)}


if __name__ == "__main__":
    cherries.run(main)
