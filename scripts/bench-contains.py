from collections.abc import Generator
from typing import no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh as tm
import warp as wp
from jaxtyping import Bool, Float
from liblaf.grapes.bench import Bencher, BenchResults

from liblaf import melon


def contains_trimesh(
    mesh: pv.PolyData, points: Float[np.ndarray, "N 3"]
) -> Bool[np.ndarray, " N"]:
    mesh_tm: tm.Trimesh = melon.as_trimesh(mesh)
    return mesh_tm.contains(points)


@wp.kernel
@no_type_check
def contains_warp_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dist: wp.float32,
    output: wp.array(dtype=wp.bool),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point(mesh_id, point, max_dist)
    output[tid] = query.sign < 0


def contains_warp(
    mesh: pv.PolyData, points: Float[np.ndarray, "N 3"]
) -> Bool[np.ndarray, " N"]:
    mesh_wp: wp.Mesh = wp.Mesh(
        wp.from_numpy(mesh.points, wp.vec3),
        wp.from_numpy(mesh.regular_faces, wp.int32).flatten(),
    )
    output_wp: wp.array = wp.zeros(points.shape[0], dtype=wp.bool)
    wp.launch(
        contains_warp_kernel,
        dim=(points.shape[0],),
        inputs=[mesh_wp.id, wp.from_numpy(points, wp.vec3), mesh.length],
        outputs=[output_wp],
    )
    return output_wp.numpy()


def main() -> None:
    mesh: pv.PolyData = pv.examples.download_bunny()  # pyright: ignore[reportAssignmentType]
    mesh.clean(inplace=True)
    mesh = melon.mesh_fix(mesh)

    bencher = Bencher(timeout=10.0, warmup=1)

    @bencher.setup
    def _() -> Generator[tuple[tuple[Float[np.ndarray, "N 3"]], dict]]:
        generator: np.random.Generator = np.random.default_rng()
        for n in np.logspace(3, 8, num=10, dtype=int):
            points: Float[np.ndarray, "N 3"] = generator.uniform(
                low=mesh.bounds[::2], high=mesh.bounds[1::2], size=(n, 3)
            )
            yield (points,), {}

    @bencher.size
    def _(points: Float[np.ndarray, "N 3"]) -> int:
        return points.shape[0]

    @bencher.bench(label="trimesh")
    def _(points: Float[np.ndarray, "N 3"]) -> Bool[np.ndarray, " N"]:
        return contains_trimesh(mesh, points)

    @bencher.bench(label="warp")
    def _(points: Float[np.ndarray, "N 3"]) -> Bool[np.ndarray, " N"]:
        return contains_warp(mesh, points)

    results: BenchResults = bencher.run()

    for label, outputs in results.outputs.items():
        if label == "trimesh":
            continue
        for size, actual, expected in zip(
            results.sizes, outputs, results.outputs["trimesh"], strict=True
        ):
            if actual is None or expected is None:
                continue
            actual: Bool[np.ndarray, " N"]
            expected: Bool[np.ndarray, " N"]
            ic(size, np.count_nonzero(actual != expected) / size)

    results.plot()
    plt.show()


if __name__ == "__main__":
    main()
