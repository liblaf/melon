from pathlib import Path

import trimesh as tm

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    input: Path = cherries.input("00-complete_human_head_anatomy.glb")
    cranium: Path = cherries.output("11-cranium.ply")
    mandible: Path = cherries.output("11-mandible.ply")


CRANIUM_GEOMETRIES: list[str] = [
    "Ethmoid_skull001_Head_skeleton_mtl_0",
    "Frontal_skull001_Head_skeleton_mtl_0",
    "Lacrimal_001left_skull001_Head_skeleton_mtl_0",
    "Lacrimal_right_skull001_Head_skeleton_mtl_0",
    "Maxilla hard palate_skull001_Head_skeleton_mtl_0",
    "Maxilla_left_skull001_Head_skeleton_mtl_0",
    "Maxilla_right_skull001_Head_skeleton_mtl_0",
    "Nasal_L_skull001_Head_skeleton_mtl_0",
    "Nasal_R_skull001_Head_skeleton_mtl_0",
    "Occipital_skull001_Head_skeleton_mtl_0",
    "Os temporale_left_skull001_Head_skeleton_mtl_0",
    "Os temporale_right_skull001_Head_skeleton_mtl_0",
    "Palatine_skull001_Head_skeleton_mtl_0",
    "Parietal_left_skull001_Head_skeleton_mtl_0",
    "Parietal_right_skull001_Head_skeleton_mtl_0",
    "Sphenoid_skull001_Head_skeleton_mtl_0",
    "Teeth Canine_029_Teeth_MTL_0",
    "Teeth Canine_030_Teeth_MTL_0",
    "Teeth Incisor central_027_Teeth_MTL_0",
    "Teeth Incisor central_028_Teeth_MTL_0",
    "Teeth Incisor lateral_028_Teeth_MTL_0",
    "Teeth Incisor lateral_029_Teeth_MTL_0",
    "Teeth Molar first_031_Teeth_MTL_0",
    "Teeth Molar first_032_Teeth_MTL_0",
    "Teeth Molar second_032_Teeth_MTL_0",
    "Teeth Molar second_033_Teeth_MTL_0",
    "Teeth Molar third_033_Teeth_MTL_0",
    "Teeth Molar third_034_Teeth_MTL_0",
    "Teeth Premolar first_029_Teeth_MTL_0",
    "Teeth Premolar first_030_Teeth_MTL_0",
    "Teeth Premolar second_030_Teeth_MTL_0",
    "Teeth Premolar second_031_Teeth_MTL_0",
    "Vomer_skull001_Head_skeleton_mtl_0",
    "Zygomatic_left_skull001_Head_skeleton_mtl_0",
    "Zygomatic_right_skull001_Head_skeleton_mtl_0",
]
MANDIBLE_GEOMETRIES: list[str] = [
    "Mandibula_skull001_Head_skeleton_mtl_0",
    "Teeth Canine_028_Teeth_MTL_0",
    "Teeth Canine_031_Teeth_MTL_0",
    "Teeth Incisor central_026_Teeth_MTL_0",
    "Teeth Incisor central_029_Teeth_MTL_0",
    "Teeth Incisor lateral_027_Teeth_MTL_0",
    "Teeth Incisor lateral_030_Teeth_MTL_0",
    "Teeth Molar first_033_Teeth_MTL_0",
    "Teeth Molar first_034_Teeth_MTL_0",
    "Teeth Molar second_034_Teeth_MTL_0",
    "Teeth Molar second_035_Teeth_MTL_0",
    "Teeth Molar third_035_Teeth_MTL_0",
    "Teeth Molar third_036_Teeth_MTL_0",
    "Teeth Premolar first_031_Teeth_MTL_0",
    "Teeth Premolar first_032_Teeth_MTL_0",
    "Teeth Premolar second_032_Teeth_MTL_0",
    "Teeth Premolar second_033_Teeth_MTL_0",
]


def main(cfg: Config) -> None:
    scene: tm.Scene = tm.load_scene(cfg.input)
    cranium: tm.Trimesh = tm.util.concatenate(
        melon.scene.extract_geometries(scene, CRANIUM_GEOMETRIES)
    )
    cranium.export(cfg.cranium)
    mandible: tm.Trimesh = tm.util.concatenate(
        melon.scene.extract_geometries(scene, MANDIBLE_GEOMETRIES)
    )
    mandible.export(cfg.mandible)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
