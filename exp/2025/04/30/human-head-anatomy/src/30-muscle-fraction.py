from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("00-Full human head anatomy.obj")
    groups: Path = cherries.input("13-groups.toml")
    tetmesh: Path = cherries.input("21-tetmesh.vtu")

    output: Path = cherries.output("30-muscle-fraction.vtu")

    n_samples: int = 1000


def main(cfg: Config) -> None:
    full: pv.PolyData = melon.load_polydata(cfg.full)
    full.clean(inplace=True)
    groups: dict[str, list[str]] = grapes.load(cfg.groups)
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)

    barycentric: Float[np.ndarray, "cells samples 4"] = melon.sample_barycentric_coords(
        (mesh.n_cells, cfg.n_samples, 4)
    )
    cells: Integer[np.ndarray, "cells 4"] = mesh.cells_dict[pv.CellType.TETRA]  # pyright: ignore[reportArgumentType]
    samples: Float[np.ndarray, "cells samples 3"] = melon.barycentric_to_points(
        mesh.points[cells][:, np.newaxis, :, :], barycentric
    )
    samples: Float[np.ndarray, "cells*samples 3"] = samples.reshape(
        mesh.n_cells * cfg.n_samples, 3
    )

    muscle_blocks: list[pv.PolyData] = []
    muscle_names: list[str] = []
    for muscle_name in groups["Muscles"]:
        muscle: pv.PolyData = melon.tri.extract_groups(full, muscle_name)
        blocks: pv.MultiBlock = muscle.split_bodies().as_polydata_blocks()
        for i, block in enumerate(blocks):
            block_name: str = f"{muscle_name}_{i:02d}"
            block: pv.PolyData = melon.ext.mesh_fix(block)  # noqa: PLW2901
            muscle_blocks.append(block)
            muscle_names.append(block_name)

    muscle_fractions_list: list[Float[np.ndarray, " cells"]] = []
    for block_name, block in grapes.track(
        zip(muscle_names, muscle_blocks, strict=True), total=len(muscle_blocks)
    ):
        ic(block_name)
        contains: Bool[np.ndarray, " cells*samples"] = melon.tri.contains(
            block, samples
        )
        contains: Bool[np.ndarray, "cells samples"] = contains.reshape(
            mesh.n_cells, cfg.n_samples
        )
        fraction: Float[np.ndarray, " cells"] = (
            np.count_nonzero(contains, axis=-1) / cfg.n_samples
        )
        muscle_fractions_list.append(fraction)
    muscle_fractions: Float[np.ndarray, "cells muscles"] = np.stack(
        muscle_fractions_list, axis=-1
    )
    major_muscle_id: Integer[np.ndarray, " cells"] = np.argmax(
        muscle_fractions, axis=-1
    )

    mesh.cell_data["MuscleFractions"] = np.max(muscle_fractions, axis=-1)
    mesh.cell_data["MuscleIds"] = major_muscle_id
    mesh.field_data["MuscleNames"] = muscle_names

    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
