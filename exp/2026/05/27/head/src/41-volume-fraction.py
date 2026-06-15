from pathlib import Path

import pyvista as pv
import torch
from jaxtyping import Float
from torch import Tensor

from liblaf import cherries, melon
from liblaf.melon.recipe.fractions import FractionResult, compute_fractions


class Config(cherries.BaseConfig):
    suffix: str = "3191k"
    muscles: Path = cherries.input("30-muscles.m.vtkhdf")
    smas: Path = cherries.input("32-smas.vtp")


def main(cfg: Config) -> None:
    torch.set_default_device("cuda")
    mesh: pv.UnstructuredGrid = melon.io.load_unstructured_grid(
        cherries.input(f"40-tetwild-{cfg.suffix}.vtu")
    )
    muscles: pv.MultiBlock = melon.io.load_multiblock(cfg.muscles)
    smas: pv.PolyData = melon.io.load_polydata(cfg.smas)

    fractions: FractionResult = compute_fractions(mesh=mesh, muscles=muscles, smas=smas)
    aponeurosis_fraction: Float[Tensor, " c"] = fractions.aponeurosis_fraction
    muscle_fraction: Float[Tensor, " c"] = fractions.muscle_fraction
    smas_fraction: Float[Tensor, " c"] = fractions.smas_fraction
    fat_fraction: Float[Tensor, " c"] = 1.0 - aponeurosis_fraction - muscle_fraction
    mesh.cell_data["AponeurosisFraction"] = aponeurosis_fraction.numpy(force=True)
    mesh.cell_data["FatFraction"] = fat_fraction.numpy(force=True)
    mesh.cell_data["MuscleFraction"] = muscle_fraction.numpy(force=True)
    mesh.cell_data["MuscleId"] = fractions.muscle_id.numpy(force=True)
    mesh.cell_data["SMASFraction"] = smas_fraction.numpy(force=True)
    mesh.field_data["MuscleName"] = muscles.keys()

    melon.save(mesh, cherries.output(f"41-tetmesh-{cfg.suffix}.vtu"))


if __name__ == "__main__":
    cherries.main(main)
