from pathlib import Path

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    input: Path = cherries.input("21-skin-thickness.vtp")
    inner: Path = cherries.output("22-smas-inner.vtp")
    outer: Path = cherries.output("22-smas-outer.vtp")


def main(cfg: Config) -> None:
    skin: pv.PolyData = melon.load_polydata(cfg.input)
    smas_inner: pv.PolyData = skin.warp_by_scalar("SkinToMuscleMaxSmooth", factor=-1.0)  # pyright: ignore[reportAssignmentType]
    smas_inner = melon.ext.mesh_fix(smas_inner)
    melon.save(cfg.inner, smas_inner)
    smas_outer: pv.PolyData = skin.warp_by_scalar("SkinToMuscleMinSmooth", factor=-1.0)  # pyright: ignore[reportAssignmentType]
    smas_outer = melon.ext.mesh_fix(smas_outer)
    melon.save(cfg.outer, smas_outer)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
