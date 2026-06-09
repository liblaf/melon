import logging
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import trimesh as tm
import warp as wp
from jaxtyping import Bool, Float
from torch import Tensor

from liblaf import cherries, melon

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    suffix: str = "3191k"
    teeth: Path = cherries.input("11-teeth.ply")
    cranium: Path = cherries.input("13-cranium.ply")
    mandible: Path = cherries.input("13-mandible.ply")
    gingiva: Path = cherries.input("20-gingiva.ply")
    skin: Path = cherries.input("22-skin.vtp")


FACE_GROUPS: list[str] = [
    "Chin",
    "EyelidBottom",
    "EyelidOuterBottom",
    "EyelidOuterTop",
    "EyelidTop",
    "Face",
    "LipBottom",
    "LipOuterBottom",
    "LipOuterTop",
    "LipTop",
]
FIXED_GROUPS: list[str] = [
    "Cranium",
    "Mandible",
]
LIP_GROUPS: list[str] = [
    "LipBottom",
    "LipInnerBottom",
    "LipInnerTop",
    "LipOuterBottom",
    "LipOuterTop",
    "LipTop",
]


def transfer_group_id(
    cranium: pv.PolyData,
    mandible: pv.PolyData,
    gingiva: pv.PolyData,
    skin: pv.PolyData,
    surface: pv.PolyData,
) -> pv.PolyData:
    group_names: list[str] = skin.field_data["GroupName"].tolist()
    group_names.extend(["Cranium", "Mandible", "Gingiva"])
    cranium.cell_data["GroupId"] = np.full(
        (cranium.n_cells,), group_names.index("Cranium"), np.int32
    )
    mandible.cell_data["GroupId"] = np.full(
        (mandible.n_cells,), group_names.index("Mandible"), np.int32
    )
    gingiva.cell_data["GroupId"] = np.full(
        (gingiva.n_cells,), group_names.index("Gingiva"), np.int32
    )
    skin.cell_data["GroupId"] = skin.cell_data["GroupId"].astype(np.int32)
    source: pv.PolyData = pv.merge([cranium, mandible, gingiva, skin])

    surface: pv.PolyData = melon.xfer.tri_cell_to_tri_point(
        source, surface, names=("GroupId",), tolerance=0.0, snap_to_closest_point=True
    )
    surface.field_data["GroupName"] = group_names
    n_invalid: np.integer = np.count_nonzero(
        surface.point_data["vtkValidPointMask"] == 0
    )
    if n_invalid > 0:
        logger.warning("# invalid points: %d / %d", n_invalid, surface.n_points)
    return surface


def mask_teeth(teeth: pv.PolyData, surface: pv.PolyData) -> pv.PolyData:
    surface.compute_implicit_distance(teeth, inplace=True)
    implicit_distance: Float[np.ndarray, " p"] = surface.point_data["implicit_distance"]
    surface.point_data["IsTeeth"] = implicit_distance < 0.002  # meters
    return surface


def mask_gingiva(gingiva: pv.PolyData, surface: pv.PolyData) -> pv.PolyData:
    surface.compute_implicit_distance(gingiva, inplace=True)
    implicit_distance: Float[np.ndarray, " p"] = surface.point_data["implicit_distance"]
    surface.point_data["IsGingiva"] = np.abs(implicit_distance) < 0.002  # meters
    return surface


def mask_skin(skin: pv.PolyData, surface: pv.PolyData) -> pv.PolyData:
    lip: pv.PolyData = melon.tri.extract_groups(skin, LIP_GROUPS)
    surface.compute_implicit_distance(lip, inplace=True)
    implicit_distance: Float[np.ndarray, " p"] = surface.point_data["implicit_distance"]
    surface.point_data["IsLip"] = np.abs(implicit_distance) < 0.002  # meters
    return surface


def make_face_convex(mesh: pv.UnstructuredGrid, surface: pv.PolyData) -> None:
    face_pv: pv.PolyData = surface.extract_points(
        np.flatnonzero(surface.point_data["IsFace"]), adjacent_cells=True
    )
    face_tm: tm.Trimesh = pv.to_trimesh(face_pv)
    convex_tm: tm.Trimesh = face_tm.convex_hull
    convex_wp: wp.Mesh = melon.io.as_warp_mesh(convex_tm)
    centers: pv.PolyData = mesh.cell_centers()
    centers: Float[Tensor, "c 3"] = torch.tensor(centers.points, dtype=torch.float32)
    in_convex: Bool[Tensor, " c"] = melon.tri.contains(convex_wp, centers)
    mesh.cell_data["InFaceConvex"] = in_convex.numpy(force=True)


def main(cfg: Config) -> None:
    torch.set_default_device("cuda")
    teeth: pv.PolyData = melon.io.load_polydata(cfg.teeth)
    cranium: pv.PolyData = melon.io.load_polydata(cfg.cranium)
    mandible: pv.PolyData = melon.io.load_polydata(cfg.mandible)
    gingiva: pv.PolyData = melon.io.load_polydata(cfg.gingiva)
    skin: pv.PolyData = melon.io.load_polydata(cfg.skin)
    mesh: pv.UnstructuredGrid = melon.io.load_unstructured_grid(
        cherries.input(f"41-tetmesh-{cfg.suffix}.vtu")
    )

    surface: pv.PolyData = mesh.extract_surface(algorithm=None)
    surface: pv.PolyData = transfer_group_id(cranium, mandible, gingiva, skin, surface)
    surface: pv.PolyData = mask_teeth(teeth, surface)
    surface: pv.PolyData = mask_gingiva(gingiva, surface)
    surface: pv.PolyData = mask_skin(skin, surface)

    surface.point_data["IsFace"] = melon.tri.select_groups(
        surface, FACE_GROUPS, preference=pv.FieldAssociation.POINT
    )
    surface.point_data["IsFixed"] = melon.tri.select_groups(
        surface, FIXED_GROUPS, preference=pv.FieldAssociation.POINT
    )
    surface.point_data["IsFixed"] &= ~(
        surface.point_data["IsTeeth"]
        | surface.point_data["IsGingiva"]
        | surface.point_data["IsLip"]
    )
    melon.save(surface, cherries.output(f"42-surface-{cfg.suffix}.vtp"))

    mesh: pv.UnstructuredGrid = melon.xfer.tri_point_to_tet_point(
        surface,
        mesh,
        {
            "GroupId": -1,
            "IsFace": False,
            "IsFixed": False,
            "IsGingiva": False,
            "IsLip": False,
            "IsTeeth": False,
        },
    )
    mesh.field_data["GroupName"] = surface.field_data["GroupName"]

    make_face_convex(mesh, surface)

    melon.save(mesh, cherries.output(f"42-tetmesh-{cfg.suffix}.vtu"))


if __name__ == "__main__":
    cherries.main(main)
