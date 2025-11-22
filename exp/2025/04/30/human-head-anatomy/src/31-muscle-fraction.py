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
    muscles: Path = cherries.input("30-muscles.vtm")
    tetmesh: Path = cherries.input("21-tetmesh.vtu")

    output: Path = cherries.output("31-muscle-fraction.vtu")

    # 1000 samples will OOM on 8GB GPU
    n_samples: int = 700


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)
    muscles: pv.MultiBlock = melon.load_multi_block(cfg.muscles)

    barycentric: Float[Array, "cells samples 4"] = melon.sample_barycentric_coords(
        (mesh.n_cells, cfg.n_samples, 4)
    )
    cells: Integer[np.ndarray, "cells 4"] = mesh.cells_dict[pv.CellType.TETRA]  # pyright: ignore[reportArgumentType]
    samples: Float[Array, "cells samples 3"] = melon.barycentric_to_points(
        mesh.points[cells][:, jnp.newaxis, :, :], barycentric
    )
    samples: Float[Array, "cells*samples 3"] = samples.reshape(
        mesh.n_cells * cfg.n_samples, 3
    )

    muscle_names: list[str] = [""] * len(muscles)
    muscle_fractions: Float[np.ndarray, "cells muscles"] = np.zeros(
        (mesh.n_cells, len(muscles))
    )
    for muscle in grapes.track(muscles, total=len(muscles)):
        muscle: pv.PolyData
        muscle_id: int = muscle.field_data["MuscleId"].item()
        muscle_name: str = muscle.field_data["MuscleName"].item()
        ic(muscle_id, muscle_name)
        contains: Bool[Array, " cells*samples"] = melon.tri.contains(muscle, samples)
        contains: Bool[Array, "cells samples"] = contains.reshape(
            mesh.n_cells, cfg.n_samples
        )
        fraction: Float[Array, " cells"] = (
            jnp.count_nonzero(contains, axis=-1) / cfg.n_samples
        )
        muscle_names[muscle_id] = muscle_name
        muscle_fractions[:, muscle_id] = fraction

    mesh.cell_data["MuscleFractions"] = np.max(muscle_fractions, axis=-1)
    mesh.cell_data["MuscleIds"] = np.argmax(muscle_fractions, axis=-1)
    mesh.field_data["MuscleNames"] = muscle_names

    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
