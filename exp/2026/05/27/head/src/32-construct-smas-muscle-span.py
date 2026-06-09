from pathlib import Path
from typing import cast

import numpy as np
import potpourri3d as pp3d
import pyvista as pv
import torch
import trimesh as tm
import warp as wp
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    # cranium: Path = cherries.input("13-cranium.ply")
    # mandible: Path = cherries.input("13-mandible.ply")
    skin: Path = cherries.input("20-skin-smoothed.ply")
    muscles: Path = cherries.input("31-muscles-smas.m.vtkhdf")
    output: Path = cherries.output("32-smas.vtp")
    output_skin: Path = cherries.output("32-skin-muscle-span.vtp")


THICKNESS_THRESHOLD: float = 0.018  # meters


def skin_to_skeletons(skin: pv.PolyData, skeletons: wp.Mesh) -> None:
    points: Float[Tensor, "p 3"] = torch.tensor(skin.points, dtype=torch.float32)
    point_normals: Float[Tensor, "p 3"] = torch.tensor(
        skin.point_normals, dtype=torch.float32
    )
    distance: Float[Tensor, " p"] = melon.tri.query_ray(
        skeletons, points, -point_normals, max_t=THICKNESS_THRESHOLD
    )
    skin.point_data["ToSkeletons"] = distance.numpy(force=True)


def skin_to_muscles(skin: pv.PolyData, muscles: wp.Mesh) -> None:
    # skin_to_skeletons: Float[np.ndarray, " p"] = skin.point_data["ToSkeletons"]
    # skin_to_skeletons: Float[Tensor, " p"] = torch.tensor(
    #     skin_to_skeletons, dtype=torch.float32
    # )
    # skin_to_skeletons.nan_to_num_(nan=THICKNESS_THRESHOLD)
    points: Float[Tensor, "p 3"] = torch.tensor(skin.points, dtype=torch.float32)
    point_normals: Float[Tensor, "p 3"] = torch.tensor(
        skin.point_normals, dtype=torch.float32
    )
    distance_min: Float[Tensor, " p"] = melon.tri.query_ray(
        muscles, points, -point_normals, max_t=THICKNESS_THRESHOLD
    )
    distance_max: Float[Tensor, " p"] = THICKNESS_THRESHOLD - melon.tri.query_ray(
        muscles,
        points - THICKNESS_THRESHOLD * point_normals,
        point_normals,
        max_t=THICKNESS_THRESHOLD,
    )
    valid: Bool[Tensor, " p"] = distance_min < distance_max
    distance_min[~valid] = torch.nan
    distance_max[~valid] = torch.nan
    skin.point_data["ToMusclesMin"] = distance_min.numpy(force=True)
    skin.point_data["ToMusclesMax"] = distance_max.numpy(force=True)


def extend_scalar(
    solver: pp3d.MeshVectorHeatSolver, data: Float[np.ndarray, " p"]
) -> Float[np.ndarray, " p"]:
    valid: Bool[np.ndarray, " p"] = np.isfinite(data)
    v_inds: Integer[np.ndarray, " q"] = np.flatnonzero(valid)
    return solver.extend_scalar(v_inds, data[v_inds])


def main(cfg: Config) -> None:
    torch.set_default_device("cuda")
    # cranium: pv.PolyData = melon.io.load_polydata(cfg.cranium)
    # mandible: pv.PolyData = melon.io.load_polydata(cfg.mandible)
    skin: pv.PolyData = melon.io.load_polydata(cfg.skin)
    muscles: pv.MultiBlock = melon.io.load_multiblock(cfg.muscles)
    # skeletons: pv.PolyData = pv.merge([cranium, mandible])

    skin.subdivide_adaptive(
        max_edge_len=0.001,  # meters
        inplace=True,
    )
    muscles: pv.UnstructuredGrid = muscles.combine()
    muscles: pv.PolyData = muscles.extract_surface(algorithm=None)
    # skeletons_wp: wp.Mesh = melon.io.as_warp_mesh(skeletons)
    muscles_wp: wp.Mesh = melon.io.as_warp_mesh(muscles)

    # skin_to_skeletons(skin, skeletons_wp)
    skin_to_muscles(skin, muscles_wp)

    solver: pp3d.MeshVectorHeatSolver = pp3d.MeshVectorHeatSolver(
        skin.points, skin.regular_faces
    )
    skin.point_data["ToMusclesMinExtend"] = extend_scalar(
        solver, skin.point_data["ToMusclesMin"]
    )
    skin.point_data["ToMusclesMaxExtend"] = extend_scalar(
        solver, skin.point_data["ToMusclesMax"]
    )
    skin.point_data["SmasThickness"] = (
        skin.point_data["ToMusclesMaxExtend"] - skin.point_data["ToMusclesMinExtend"]
    )
    melon.save(skin, cfg.output_skin)

    smas_outer: pv.PolyData = skin.warp_by_scalar(
        "ToMusclesMinExtend", factor=-1.0, inplace=False
    )
    smas_outer: pv.PolyData = melon.ext.meshfix(smas_outer)
    smas_inner: pv.PolyData = skin.warp_by_scalar(
        "ToMusclesMaxExtend", factor=-1.0, inplace=False
    )
    smas_inner: pv.PolyData = melon.ext.meshfix(smas_inner)
    smas_outer_tm: tm.Trimesh = pv.to_trimesh(smas_outer)
    smas_inner_tm: tm.Trimesh = pv.to_trimesh(smas_inner)
    smas_tm: tm.Trimesh = tm.boolean.difference([smas_outer_tm, smas_inner_tm])
    smas: pv.PolyData = cast("pv.PolyData", pv.wrap(smas_tm))
    smas: pv.PolyData = melon.ext.meshfix(smas)

    melon.save(smas, cfg.output)


if __name__ == "__main__":
    cherries.main(main)
