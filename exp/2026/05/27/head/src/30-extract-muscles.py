import logging
from pathlib import Path
from typing import cast

import pyvista as pv
import trimesh as tm

from liblaf import cherries, melon

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    input: Path = cherries.input("00-complete_human_head_anatomy.glb")
    output: Path = cherries.output("30-muscles.m.vtkhdf")


def main(cfg: Config) -> None:
    scene: tm.Scene = tm.load_scene(cfg.input)
    muscles_scene: tm.Scene = melon.scene.subscene(scene, "Muscles Subgroup_004")

    muscles: pv.MultiBlock = pv.MultiBlock()
    for muscle_tm in melon.scene.dump(muscles_scene, include_visual=False):
        muscle_name: str = muscle_tm.metadata["name"]
        muscle_pv: pv.PolyData = cast("pv.PolyData", pv.wrap(muscle_tm))
        muscle_pv.clean(inplace=True)
        bodies: pv.MultiBlock = muscle_pv.split_bodies()
        bodies: pv.MultiBlock = bodies.as_polydata_blocks()
        for idx, body_pv_ in enumerate(bodies):
            if len(bodies) == 1:
                body_name: str = muscle_name
            else:
                body_name: str = f"{muscle_name}_{idx:01d}"
            body_pv: pv.PolyData = cast("pv.PolyData", body_pv_)
            body_pv: pv.PolyData = melon.ext.meshfix(body_pv)
            muscles.append(body_pv, name=body_name)

    melon.save(muscles, cfg.output)


if __name__ == "__main__":
    cherries.main(main)
