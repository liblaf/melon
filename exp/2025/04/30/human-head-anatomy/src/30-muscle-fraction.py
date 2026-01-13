from pathlib import Path

import numpy as np
import pyvista as pv
from environs import env
from jaxtyping import Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes

SUFFIX: str = env.str("SUFFIX", default="-515k")


class Config(cherries.BaseConfig):
    suffix: str = SUFFIX
    full: Path = cherries.input("00-Full human head anatomy.obj")
    groups: Path = cherries.input("13-groups.toml")
    muscles: Path = cherries.input("20-muscles.vtm")
    tetmesh: Path = cherries.input(f"21-tetgen{SUFFIX}.vtu")

    output: Path = cherries.output(f"30-muscle-fraction{SUFFIX}.vtu")

    n_samples: int = 1000


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)
    muscles: pv.MultiBlock = melon.load_multi_block(cfg.muscles)

    muscle_names: list[str] = [""] * len(muscles)
    muscle_fractions: Float[np.ndarray, "muscles cells"] = np.zeros(
        (len(muscles), mesh.n_cells)
    )
    for muscle in grapes.track(muscles, total=len(muscles), description="Muscles"):
        muscle: pv.PolyData
        muscle_id: int = muscle.field_data["MuscleId"].item()
        muscle_name: str = muscle.field_data["MuscleName"].item()
        ic(muscle_id, muscle_name)
        muscle_names[muscle_id] = muscle_name
        muscle_fractions[muscle_id] = melon.tet.compute_volume_fraction(
            mesh, muscle, n_samples=cfg.n_samples
        )

    mesh.cell_data["MuscleFraction"] = np.max(muscle_fractions, axis=0)
    mesh.cell_data["MuscleId"] = np.argmax(muscle_fractions, axis=0)
    mesh.field_data["MuscleName"] = muscle_names

    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
