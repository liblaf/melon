from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Bool

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes

CRANIUM_GROUPS: list[str] = [
    "Ethmoid_skull001",
    "Frontal_skull001",
    "Lateral_cartilage_left_skull001",
    "Lateral_cartilage_right_skull001",
    "Maxilla_left_skull001",
    "Maxilla_right_skull001",
    "Nasal_L_skull001",
    "Nasal_R_skull001",
    "Occipital_skull001",
    "Os_temporale_left_skull001",
    "Os_temporale_right_skull001",
    "Parietal_left_skull001",
    "Parietal_right_skull001",
    "Septal_cartilage_skull001",
    "Sphenoid_skull001",
    "Zygomatic_left_skull001",
    "Zygomatic_right_skull001",
]


class Config(cherries.BaseConfig):
    groups: Path = Path("data/01_raw/groups.toml")
    input: Path = Path("data/01_raw/human-head-anatomy.obj")
    output: Path = Path("data/02_intermediate/")


@cherries.main()
def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.input)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)

    cranium_bodies: list[pv.PolyData] = []
    for skeleton_name in CRANIUM_GROUPS:
        selection: Bool[np.ndarray, " C"] = melon.select_cells_by_group(
            full, skeleton_name
        )
        skeleton: pv.PolyData = melon.extract_cells(full, selection)
        skeleton_fixed: pv.PolyData = melon.mesh_fix(skeleton)
        cranium_bodies.append(skeleton_fixed)
    cranium: tm.Trimesh = tm.boolean.union(
        [melon.as_trimesh(skeleton) for skeleton in cranium_bodies]
    )
    cranium = cranium.process(validate=True)
    assert melon.is_volume(cranium)
    melon.save(cfg.output / "cranium.ply", cranium)

    mandible: pv.PolyData = melon.extract_cells(
        full, melon.select_cells_by_group(full, "Mandibula_skull001")
    )
    # assert melon.is_volume(mandible)
    melon.save(cfg.output / "mandible.ply", mandible)


if __name__ == "__main__":
    main(Config())
