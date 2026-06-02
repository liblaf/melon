from pathlib import Path

import pyvista as pv
import trimesh as tm

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    input: Path = cherries.input("00-complete_human_head_anatomy.glb")
    eye: Path = cherries.output("20-eye.ply")
    gingiva: Path = cherries.output("20-gingiva.ply")
    skin: Path = cherries.output("20-skin.ply")
    skin_smoothed: Path = cherries.output("20-skin-smoothed.ply")


EYE_GEOMETRIES: list[str] = [
    "Eye L001_Eyes_mtl_0",
    "Eye R001_Eyes_mtl_0",
]
GINGIVA_GEOMETRIES: list[str] = [
    "Gingiva001_Oral_cavity_mtl_0",
]
SKIN_GEOMETRIES: list[str] = [
    "skin Cross section_Head_Neck_mtl_0",
    "Skin001_Head_Neck_mtls_0",
]


def main(cfg: Config) -> None:
    scene: tm.Scene = tm.load_scene(cfg.input)
    eye: tm.Trimesh = tm.util.concatenate(
        melon.scene.extract_geometries(scene, EYE_GEOMETRIES)
    )
    eye.export(cfg.eye)
    gingiva: tm.Trimesh = tm.util.concatenate(
        melon.scene.extract_geometries(scene, GINGIVA_GEOMETRIES)
    )
    gingiva.export(cfg.gingiva)
    skin: tm.Trimesh = tm.util.concatenate(
        melon.scene.extract_geometries(scene, SKIN_GEOMETRIES)
    )
    skin_pv: pv.PolyData = melon.io.as_polydata(skin)
    # trimesh.Trimesh.merge_vertices() does not work well
    skin_pv.clean(tolerance=0.5 * melon.tri.edge_length(skin_pv).min(), inplace=True)
    skin_pv.clear_data()
    melon.save(skin_pv, cfg.skin)
    skin_pv.smooth_taubin(inplace=True)
    melon.save(skin_pv, cfg.skin_smoothed)


if __name__ == "__main__":
    cherries.main(main)
