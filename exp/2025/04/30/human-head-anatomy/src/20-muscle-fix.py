from pathlib import Path

import pyvista as pv
import trimesh as tm

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("00-Full human head anatomy.obj")
    groups: Path = cherries.input("13-groups.toml")
    skin: Path = cherries.input("12-skin.vtp")

    output: Path = cherries.output("20-muscles.vtm")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.full)
    full.clean(inplace=True)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    skin: pv.PolyData = melon.load_polydata(cfg.skin)
    face: pv.PolyData = melon.tri.extract_groups(
        skin,
        [
            "Ear",
            "EarNeckBack",
            "EarSocket",
            "EyeSocketBottom",
            "EyeSocketTop",
            "HeadBack",
            "LipInnerBottom",
            "LipInnerTop",
            "MouthSocketBottom",
            "MouthSocketTop",
            "NeckBack",
            "NeckFront",
            "Nostril",
        ],
        invert=True,
    )
    melon.save(cherries.temp("20-face.vtp"), face)
    face_convex: tm.Trimesh = tm.convex.convex_hull(face.points)
    melon.save(cherries.temp("20-face-convex.ply"), face_convex)

    block_id: int = 0
    muscles: pv.MultiBlock = pv.MultiBlock()
    for muscle_name in groups["Muscles"]:
        muscle: pv.PolyData = melon.tri.extract_groups(full, muscle_name)
        blocks: pv.MultiBlock = muscle.split_bodies().as_polydata_blocks()
        for i, block in enumerate(blocks):
            block: pv.PolyData = melon.mesh_fix(block)  # noqa: PLW2901
            if not face_convex.contains(block.points).any():
                continue
            block_id: int = len(muscles)
            block_name: str = (
                f"{muscle_name}_{i:02d}" if len(blocks) > 1 else muscle_name
            )
            ic(block_id, block_name)
            block.field_data["MuscleId"] = block_id  # pyright: ignore[reportArgumentType]
            block.field_data["MuscleName"] = block_name
            muscles.append(block, name=block_name)
    melon.save(cfg.output, muscles)


if __name__ == "__main__":
    cherries.main(main)
