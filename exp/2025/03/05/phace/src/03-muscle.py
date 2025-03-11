import concurrent.futures
import itertools
from pathlib import Path

import numpy as np
import pyvista as pv
import rich.progress
import trimesh as tm
from jaxtyping import Float

import liblaf.cherries as cherries  # noqa: PLR0402
import liblaf.grapes as grapes  # noqa: PLR0402
from liblaf import melon


class Config(cherries.BaseConfig):
    raw: Path = Path("./data/00-raw/")
    intermediate: Path = Path("./data/01-intermediate/")


@cherries.main()
def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(
        cfg.intermediate / "01-segmented.vtu"
    )
    muscles: list[pv.PolyData] = load_muscles(cfg.raw / "muscles")
    mesh.cell_data["muscle-fraction"] = np.zeros((mesh.n_cells, len(muscles)))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for muscle_id, muscle in enumerate(muscles):
            muscle_tm: tm.Trimesh = melon.as_trimesh(muscle)
            with grapes.progress() as progress:
                task_id: rich.progress.TaskID = progress.add_task(
                    f"Muscle {muscle_id}", total=mesh.n_cells
                )
                for cell_id, muscle_fraction in enumerate(
                    executor.map(
                        compute_muscle_fraction,
                        itertools.repeat(muscle_tm),
                        mesh.points[mesh.cells_dict[pv.CellType.TETRA]],
                        chunksize=32,
                    )
                ):
                    mesh.cell_data["muscle-fraction"][cell_id][muscle_id] = (
                        muscle_fraction
                    )
                    progress.advance(task_id)
    mesh.cell_data["orientation"] = np.full((mesh.n_cells, 3), np.nan)
    for cell_id in range(mesh.n_cells):
        muscle_id: int = np.argmax(mesh.cell_data["muscle-fraction"][cell_id])  # pyright: ignore[reportAssignmentType]
        if mesh.cell_data["muscle-fraction"][cell_id][muscle_id] < 1e-3:
            continue
        mesh.cell_data["orientation"][cell_id] = muscles[muscle_id].field_data[
            "orientation"
        ]
    melon.save(cfg.intermediate / "03-muscle.vtu", mesh)


def load_muscles(dpath: Path) -> list[pv.PolyData]:
    muscles: list[pv.PolyData] = [
        melon.load_poly_data(fpath) for fpath in dpath.glob("*.vtp")
    ]
    return muscles


def compute_muscle_fraction(
    muscle: tm.Trimesh, cell: Float[np.ndarray, "4 3"]
) -> float:
    cell: tm.Trimesh = tetra_to_trimesh(cell)
    cell = cell.process(validate=True)
    intersection: tm.Trimesh = tm.boolean.intersection([muscle, cell])
    return intersection.volume / cell.volume


def tetra_to_trimesh(tetra: Float[np.ndarray, "4 3"]) -> tm.Trimesh:
    mesh = tm.Trimesh(
        vertices=tetra,
        faces=list(itertools.permutations(range(4), 3)),
        process=True,
        validate=True,
    )
    return mesh


if __name__ == "__main__":
    main(Config())
