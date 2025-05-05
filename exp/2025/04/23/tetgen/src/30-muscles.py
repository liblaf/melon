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
import liblaf.melon as melon  # noqa: PLR0402


class Config(cherries.BaseConfig):
    full: Path = Path("data/01-raw/Full human head anatomy.obj")
    groups: Path = Path("data/02-intermediate/groups.toml")
    input: Path = Path("data/03-primary/tetgen-clean.vtu")
    output: Path = Path("data/03-primary/tetgen-clean-muscle.vtu")


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_poly_data(cfg.full)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    # muscles: dict[str, pv.PolyData] = load_muscles(full, groups["Muscles"])
    muscles: dict[str, pv.PolyData] = load_muscles(
        full,
        ["Levator_labii_superioris001", "Zygomaticus_minor001", "Zygomaticus_major001"],
    )
    for muscle_name in muscles:
        mesh.cell_data[muscle_name] = np.zeros((mesh.n_cells,))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for muscle_name, muscle in muscles.items():
            muscle_tm: tm.Trimesh = melon.as_trimesh(muscle)
            with grapes.progress() as progress:
                task_id: rich.progress.TaskID = progress.add_task(
                    muscle_name, total=mesh.n_cells
                )
                for cell_id, muscle_fraction in enumerate(
                    executor.map(
                        compute_muscle_fraction,
                        itertools.repeat(muscle_tm),
                        mesh.points[mesh.cells_dict[pv.CellType.TETRA]],
                        chunksize=32,
                    )
                ):
                    mesh.cell_data[muscle_name][cell_id] = muscle_fraction
                    progress.advance(task_id)
    mesh.cell_data["muscle-fraction"] = np.zeros((mesh.n_cells,))
    for muscle_name in muscles:
        mesh.cell_data["muscle-fraction"] += mesh.cell_data[muscle_name]
    melon.save(cfg.output, mesh)


def load_muscles(full: pv.PolyData, groups: list[str]) -> dict[str, pv.PolyData]:
    muscles: dict[str, pv.PolyData] = {}
    for group in groups:
        muscle: pv.PolyData = melon.triangle.extract_groups(full, group)
        blocks: pv.MultiBlock = muscle.split_bodies().as_polydata_blocks()
        sub_muscles: list[pv.PolyData] = []
        for block in blocks:
            sub_muscle: pv.PolyData = melon.plugin.mesh_fix(block)
            sub_muscles.append(sub_muscle)
        muscles[group] = pv.merge(sub_muscles)
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
    cherries.run(main)
