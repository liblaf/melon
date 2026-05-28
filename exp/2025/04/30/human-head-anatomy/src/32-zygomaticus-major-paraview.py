from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Float

import liblaf.melon as melon  # noqa: PLR0402

ACTIVATION: Float[np.ndarray, " 6"] = np.asarray(
    [-0.5, 0.2, 0.1, 0.0, 0.0, 0.0], dtype=np.float32
)
AXIS_NAMES: tuple[str, str, str] = ("extension", "width", "thickness")
OUTPUT_NAME: str = "32-zygomaticus-major-orientation-deformation.vtm"


def main() -> None:
    exp_dir: Path = Path(__file__).resolve().parents[1]
    muscles: pv.MultiBlock = melon.load_multi_block(exp_dir / "data/30-muscles.vtm")
    scene: pv.MultiBlock = zygomaticus_major_scene(muscles, ACTIVATION)
    output: Path = exp_dir / "data" / OUTPUT_NAME
    melon.save(output, scene)
    print(output)


def zygomaticus_major_scene(
    muscles: pv.MultiBlock, activation: Float[np.ndarray, " 6"]
) -> pv.MultiBlock:
    scene: pv.MultiBlock = pv.MultiBlock()
    for muscle in select_muscles(muscles, "Zygomaticus_major001"):
        name: str = muscle_name(muscle)
        orientation: Float[np.ndarray, "3 3"] = inertia_orientation(muscle)

        original: pv.PolyData = annotate_muscle(
            muscle.copy(deep=True),
            orientation=orientation,
            activation=activation,
            state_id=0,
        )
        deformed: pv.PolyData = annotate_muscle(
            deform_muscle(muscle, orientation, activation),
            orientation=orientation,
            activation=activation,
            state_id=1,
        )

        scene.append(original, name=f"original_{name}")
        scene.append(deformed, name=f"deformed_{name}")
        add_axis_arrows(scene, muscle, original, orientation, state_id=0)
        add_axis_arrows(scene, muscle, deformed, orientation, state_id=1)
    return scene


def select_muscles(muscles: pv.MultiBlock, prefix: str) -> list[pv.PolyData]:
    selected: list[pv.PolyData] = []
    for muscle in muscles:
        muscle: pv.PolyData
        if muscle_name(muscle).startswith(prefix):
            selected.append(muscle)
    return sorted(selected, key=muscle_id)


def muscle_id(muscle: pv.PolyData) -> int:
    return int(muscle.field_data["MuscleId"].item())


def muscle_name(muscle: pv.PolyData) -> str:
    return str(np.asarray(muscle.field_data["MuscleName"]).item())


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
    width: Float[np.ndarray, " 3"] = axes[1]
    thickness: Float[np.ndarray, " 3"] = np.cross(extension, width)
    thickness /= np.linalg.norm(thickness)
    return np.asarray([extension, width, thickness], dtype=np.float32)


def activation_scales(activation: Float[np.ndarray, " 6"]) -> Float[np.ndarray, " 3"]:
    return np.asarray(1.0 + activation[:3], dtype=np.float32)


def deform_muscle(
    muscle: pv.PolyData,
    orientation: Float[np.ndarray, "3 3"],
    activation: Float[np.ndarray, " 6"],
) -> pv.PolyData:
    original_points: Float[np.ndarray, "points 3"] = np.asarray(muscle.points)
    center: Float[np.ndarray, " 3"] = np.asarray(muscle.center_of_mass())
    local: Float[np.ndarray, "points 3"] = (original_points - center) @ orientation.T
    deformed_points: Float[np.ndarray, "points 3"] = (
        center + (local * activation_scales(activation)) @ orientation
    )

    deformed: pv.PolyData = muscle.copy(deep=True)
    deformed.points = deformed_points
    deformed.point_data["Displacement"] = deformed_points - original_points
    return deformed


def annotate_muscle(
    muscle: pv.PolyData,
    orientation: Float[np.ndarray, "3 3"],
    activation: Float[np.ndarray, " 6"],
    state_id: int,
) -> pv.PolyData:
    center: Float[np.ndarray, " 3"] = np.asarray(muscle.center_of_mass())
    muscle.field_data["MuscleOrientation"] = orientation.ravel()
    muscle.field_data["Activation"] = activation
    muscle.field_data["ActivationScale"] = activation_scales(activation)
    muscle.point_data["LocalCoordinate"] = (np.asarray(muscle.points) - center) @ (
        orientation.T
    )
    muscle.cell_data["MuscleId"] = np.full(
        muscle.n_cells, muscle_id(muscle), dtype=np.int32
    )
    muscle.cell_data["StateId"] = np.full(muscle.n_cells, state_id, dtype=np.int32)
    return muscle


def add_axis_arrows(
    scene: pv.MultiBlock,
    source_muscle: pv.PolyData,
    display_muscle: pv.PolyData,
    orientation: Float[np.ndarray, "3 3"],
    state_id: int,
) -> None:
    scales: Float[np.ndarray, " 3"] = (
        activation_scales(ACTIVATION) if state_id == 1 else np.ones(3)
    )
    state: str = "original" if state_id == 0 else "deformed"
    center: Float[np.ndarray, " 3"] = np.asarray(display_muscle.center_of_mass())
    for axis_id, axis_name in enumerate(AXIS_NAMES):
        arrow: pv.PolyData = pv.Arrow(
            start=center,
            direction=orientation[axis_id],
            scale=float(0.16 * source_muscle.length * scales[axis_id]),
            tip_radius=0.08,
            shaft_radius=0.025,
        )
        arrow.field_data["MuscleName"] = muscle_name(source_muscle)
        arrow.field_data["AxisName"] = axis_name
        arrow.cell_data["MuscleId"] = np.full(
            arrow.n_cells, muscle_id(source_muscle), dtype=np.int32
        )
        arrow.cell_data["AxisId"] = np.full(arrow.n_cells, axis_id, dtype=np.int32)
        arrow.cell_data["StateId"] = np.full(arrow.n_cells, state_id, dtype=np.int32)
        scene.append(arrow, name=f"{axis_name}_axis_{state}_{muscle_name(source_muscle)}")


if __name__ == "__main__":
    main()
