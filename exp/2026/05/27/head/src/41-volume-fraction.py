from pathlib import Path

import pyvista as pv
import torch

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
    mesh.cell_data["AponeurosisFraction"] = fractions.aponeurosis_fraction.numpy(
        force=True
    )
    mesh.cell_data["MuscleFraction"] = fractions.muscle_fraction.numpy(force=True)
    mesh.cell_data["MuscleId"] = fractions.muscle_id.numpy(force=True)
    mesh.cell_data["SMASFraction"] = fractions.smas_fraction.numpy(force=True)
    mesh.field_data["MuscleName"] = muscles.keys()

    melon.save(mesh, cherries.output(f"41-tetmesh-{cfg.suffix}.vtu"))


if __name__ == "__main__":
    cherries.main(main)
