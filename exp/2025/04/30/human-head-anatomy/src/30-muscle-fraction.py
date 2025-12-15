import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".75"


import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Bool, Float, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    full: Path = cherries.input("00-Full human head anatomy.obj")
    groups: Path = cherries.input("13-groups.toml")
    muscles: Path = cherries.input("20-muscles.vtm")
    tetmesh: Path = cherries.input("21-tetgen-68k-coarse.vtu")

    output: Path = cherries.output("30-muscle-fraction-68k-coarse.vtu")

    n_samples: int = 1000


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)
    muscles: pv.MultiBlock = melon.load_multi_block(cfg.muscles)

    cells: Integer[Array, "cells 4"] = jnp.asarray(mesh.cells_dict[pv.CellType.TETRA])  # pyright: ignore[reportArgumentType]

    muscle_names: list[str] = [""] * len(muscles)
    muscle_fractions: Float[np.ndarray, "cells muscles"] = np.zeros(
        (mesh.n_cells, len(muscles))
    )
    for muscle in grapes.track(muscles, total=len(muscles), description="Muscles"):
        muscle: pv.PolyData
        muscle_id: int = muscle.field_data["MuscleId"].item()
        muscle_name: str = muscle.field_data["MuscleName"].item()
        ic(muscle_id, muscle_name)
        muscle_names[muscle_id] = muscle_name

        solver = melon.tri.MeshContainsPoints(muscle)

        point_contains: Bool[Array, " points"] = solver.contains(mesh.points)
        cell_all_in: Bool[Array, " cells"] = jnp.all(point_contains[cells], axis=-1)
        cell_all_out: Bool[Array, " cells"] = jnp.all(~point_contains[cells], axis=-1)
        muscle_fractions[cell_all_in, muscle_id] = 1.0
        muscle_fractions[cell_all_out, muscle_id] = 0.0
        cell_cross: Integer[Array, " cells"] = jnp.flatnonzero(
            ~(cell_all_in | cell_all_out)
        )
        ic(mesh.n_cells, cell_cross.size)
        if cell_cross.size == 0:
            continue

        for chunk in jnp.array_split(
            cell_cross, max(math.floor(cell_cross.size * cfg.n_samples // 10**7), 1)
        ):
            barycentric: Float[Array, "cells samples 4"] = (
                melon.sample_barycentric_coords((chunk.size, cfg.n_samples, 4))
            )
            samples: Float[Array, "cells samples 3"] = melon.barycentric_to_points(
                mesh.points[cells[chunk]][:, jnp.newaxis, :, :],
                barycentric,
            )
            contains: Bool[Array, "cells samples"] = solver.contains(
                samples.reshape(chunk.size * cfg.n_samples, 3)
            ).reshape(chunk.size, cfg.n_samples)
            fraction: Float[Array, " cells"] = (
                jnp.count_nonzero(contains, axis=-1) / cfg.n_samples
            )
            muscle_fractions[chunk, muscle_id] = fraction

    mesh.cell_data["MuscleFraction"] = np.max(  # pyright: ignore[reportArgumentType]
        muscle_fractions, axis=-1
    )
    mesh.cell_data["MuscleId"] = np.argmax(  # pyright: ignore[reportArgumentType]
        muscle_fractions, axis=-1
    )
    mesh.field_data["MuscleName"] = muscle_names

    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
