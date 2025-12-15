from pathlib import Path
from typing import cast

import pyvista as pv

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries
from liblaf.melon import compat


class Config(cherries.BaseConfig):
    skin: Path = cherries.input("00-XYZ_ReadyToSculpt_eyesOpen_PolyGroups_GEO.obj")


def main(cfg: Config) -> None:
    skin: pv.PolyData = melon.load_polydata(cfg.skin)
    skin.clean(inplace=True)
    skin.field_data["GroupName"] = [
        cast("str", name).split(maxsplit=1)[0] for name in compat.get_group_name(skin)
    ]
    face: pv.PolyData = melon.tri.extract_groups(
        skin,
        [
            "Ear",
            "EarNeckBack",
            "EarSocket",
            "EyeSocketBottom",
            "EyeSocketTop",
            "HeadBack",
            # "LipInnerBottom",
            # "LipInnerTop",
            "MouthSocketBottom",
            "MouthSocketTop",
            "NeckBack",
            "NeckFront",
            "Nostril",
        ],
        invert=True,
    )
    melon.save(cherries.temp("10-face.vtp"), face)
    ic(face)
    ic(melon.compute_edges_length(face).mean() / skin.length)


if __name__ == "__main__":
    cherries.main(main)
