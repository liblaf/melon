from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Bool, Float, Integer

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries, grapes


class Config(cherries.BaseConfig):
    muscles: Path = cherries.input("30-muscles.vtm")
    expression_515k: Path = cherries.input("41-expression-515k.vtu")
    expression_3152k: Path = cherries.input("41-expression-3152k.vtu")
    output_515k: Path = cherries.output("42-expression-muscle-orientation-515k.vtu")
    output_3152k: Path = cherries.output("42-expression-muscle-orientation-3152k.vtu")


def main(cfg: Config) -> None:
    muscles: pv.MultiBlock = melon.load_multi_block(cfg.muscles)
    orientations: Float[np.ndarray, "muscles 9"] = muscle_orientations(muscles)

    for input_path, output_path in [
        (cfg.expression_515k, cfg.output_515k),
        (cfg.expression_3152k, cfg.output_3152k),
    ]:
        mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(input_path)
        add_muscle_orientations(mesh, orientations)
        melon.save(output_path, mesh)


def muscle_orientations(muscles: pv.MultiBlock) -> Float[np.ndarray, "muscles 9"]:
    orientations: Float[np.ndarray, "muscles 9"] = np.zeros(
        (len(muscles), 9), dtype=np.float32
    )
    for muscle in grapes.track(muscles, total=len(muscles), description="Muscles"):
        muscle: pv.PolyData
        muscle_id: int = int(muscle.field_data["MuscleId"].item())
        orientations[muscle_id] = inertia_orientation(muscle).ravel()
    return orientations


def inertia_orientation(muscle: pv.PolyData) -> Float[np.ndarray, "3 3"]:
    muscle_tm: tm.Trimesh = melon.as_trimesh(muscle)
    components: Float[np.ndarray, " 3"] = np.asarray(
        muscle_tm.principal_inertia_components
    )
    vectors: Float[np.ndarray, "3 3"] = np.asarray(
        muscle_tm.principal_inertia_vectors
    )
    axes: Float[np.ndarray, "3 3"] = np.asarray(
        vectors[np.argsort(components)], dtype=np.float64
    )
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)

    extension: Float[np.ndarray, " 3"] = axes[0]
    transverse: Float[np.ndarray, " 3"] = axes[1]
    thickness: Float[np.ndarray, " 3"] = np.cross(extension, transverse)
    thickness /= np.linalg.norm(thickness)
    return np.asarray([extension, transverse, thickness], dtype=np.float32)


def add_muscle_orientations(
    mesh: pv.UnstructuredGrid, orientations: Float[np.ndarray, "muscles 9"]
) -> None:
    muscle_fraction: Float[np.ndarray, " cells"] = np.asarray(
        mesh.cell_data["MuscleFraction"]
    )
    muscle_id: Integer[np.ndarray, " cells"] = np.asarray(
        mesh.cell_data["MuscleId"], dtype=np.int64
    )
    is_muscle: Bool[np.ndarray, " cells"] = (
        (muscle_fraction > 0.0) & (muscle_id >= 0) & (muscle_id < len(orientations))
    )

    cell_orientations: Float[np.ndarray, "cells 9"] = np.zeros(
        (mesh.n_cells, 9), dtype=np.float32
    )
    cell_orientations[is_muscle] = orientations[muscle_id[is_muscle]]
    mesh.cell_data["MuscleOrientation"] = cell_orientations


def activation_tensor_world(
    orientation: Float[np.ndarray, "3 3"] | Float[np.ndarray, " 9"],
    activation: Float[np.ndarray, " 6"],
) -> Float[np.ndarray, "3 3"]:
    axes: Float[np.ndarray, "3 3"] = np.reshape(orientation, (3, 3))
    return axes.T @ voigt_to_tensor(activation) @ axes


def activation_voigt_world(
    orientation: Float[np.ndarray, "3 3"] | Float[np.ndarray, " 9"],
    activation: Float[np.ndarray, " 6"],
) -> Float[np.ndarray, " 6"]:
    tensor: Float[np.ndarray, "3 3"] = activation_tensor_world(
        orientation, activation
    )
    return tensor_to_voigt(tensor)


def voigt_to_tensor(voigt: Float[np.ndarray, " 6"]) -> Float[np.ndarray, "3 3"]:
    return np.asarray(
        [
            [voigt[0], voigt[5], voigt[4]],
            [voigt[5], voigt[1], voigt[3]],
            [voigt[4], voigt[3], voigt[2]],
        ]
    )


def tensor_to_voigt(tensor: Float[np.ndarray, "3 3"]) -> Float[np.ndarray, " 6"]:
    return np.asarray(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            tensor[1, 2],
            tensor[0, 2],
            tensor[0, 1],
        ]
    )


if __name__ == "__main__":
    cherries.main(main)
